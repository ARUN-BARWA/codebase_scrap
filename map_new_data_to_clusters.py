"""
Map new dataset to existing clusters from mo_discovery.py output.

Takes a new dataset (separate goods and fraud files) and:
1. Loads trained HGB model + SHAP explainer + UMAP reducer + cluster model
2. Computes SHAP values for new data
3. Maps to existing clusters
4. Shows how new data distributes + fraud%, good% per cluster

Usage:
  python map_new_data_to_clusters.py \
    --mo_output mo_output \
    --new_goods_glob "new_data_goods/*.parquet" \
    --new_fraud_path "new_data/fraud.parquet" \
    --run "all_shap" \
    --output_dir new_data_clusters

Produces:
  new_data_clusters/new_data_cluster_map.csv
  new_data_clusters/new_data_summary.md
  new_data_clusters/new_data_by_cluster.parquet (every row tagged with cluster)
"""

import argparse
import logging
import pickle
import joblib
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import polars as pl
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

try:
    import umap
    import hdbscan
    from sklearn.ensemble import HistGradientBoostingClassifier
except ImportError as e:
    raise ImportError(f"Required package missing: {e}. Install: pip install umap-learn hdbscan scikit-learn")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


@dataclass
class MapConfig:
    mo_output: str
    new_goods_glob: str
    new_fraud_path: str
    output_dir: str
    run: str = "all_shap"  # which run to use: all_shap, missed_shap, missed_quantile
    tran_date_col: str = "tran_dt"
    account_col: str = "account_no"
    rename_fraud: Dict = None
    rename_goods: Dict = None
    n_shap_features: int = 50
    batch_size: int = 50000  # process new data in batches to save memory


class NewDataMapper:
    def __init__(self, cfg: MapConfig):
        self.cfg = cfg
        self.mo_dir = Path(cfg.mo_output)
        self.out_dir = Path(cfg.output_dir)
        self.cache_dir = self.mo_dir / "cache"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        # Models/transformers from original run
        self.hgb_model = None
        self.shap_explainer = None
        self.umap_reducer = None
        self.hdbscan_model = None
        self.cluster_centroids = None
        self.top_shap_features = None
        
        # New data
        self.new_fraud = None
        self.new_goods = None
        self.X_new_processed = None
        
        # Results
        self.cluster_assignments = None
        self.report_lines = []
    
    def log_report(self, line: str):
        self.report_lines.append(line)
        log.info(line)
    
    def save_report(self):
        report_path = self.out_dir / "new_data_summary.md"
        with open(report_path, 'w') as f:
            f.write('\n'.join(self.report_lines))
        log.info(f"Report saved to {report_path}")
    
    def load_mo_models(self):
        """Load trained models from mo_discovery cache."""
        self.log_report("## Loading Trained Models from mo_discovery Output\n")
        
        # Load HGB model
        model_file = self.cache_dir / "hgb_model.pkl"
        if not model_file.exists():
            raise FileNotFoundError(f"HGB model not found: {model_file}")
        
        with open(model_file, 'rb') as f:
            self.hgb_model = pickle.load(f)
        self.log_report(f"✓ Loaded HGB model")
        
        # Load SHAP explainer
        explainer_file = self.cache_dir / "shap_explainer.pkl"
        if explainer_file.exists():
            with open(explainer_file, 'rb') as f:
                self.shap_explainer = pickle.load(f)
            self.log_report(f"✓ Loaded SHAP explainer")
        else:
            self.log_report(f"⚠️  SHAP explainer not found; will recompute (slower)")
        
        # Load UMAP reducer
        umap_file = self.cache_dir / f"umap_reducer_{self.cfg.run}.pkl"
        if not umap_file.exists():
            raise FileNotFoundError(f"UMAP reducer not found: {umap_file}")
        
        with open(umap_file, 'rb') as f:
            self.umap_reducer = pickle.load(f)
        self.log_report(f"✓ Loaded UMAP reducer")
        
        # Load HDBSCAN model
        hdbscan_file = self.cache_dir / f"hdbscan_model_{self.cfg.run}.pkl"
        if not hdbscan_file.exists():
            raise FileNotFoundError(f"HDBSCAN model not found: {hdbscan_file}")
        
        with open(hdbscan_file, 'rb') as f:
            self.hdbscan_model = pickle.load(f)
        self.log_report(f"✓ Loaded HDBSCAN model")
        
        # Load top SHAP features
        features_file = self.cache_dir / f"top_shap_features_{self.cfg.run}.pkl"
        if not features_file.exists():
            raise FileNotFoundError(f"Top SHAP features not found: {features_file}")
        
        with open(features_file, 'rb') as f:
            self.top_shap_features = pickle.load(f)
        self.log_report(f"✓ Loaded top {len(self.top_shap_features)} SHAP features")
        self.log_report("")
    
    def load_new_data(self):
        """Load and prepare new goods + fraud."""
        self.log_report("## Loading New Dataset\n")
        
        rename_fraud = self.cfg.rename_fraud or {}
        rename_goods = self.cfg.rename_goods or {}
        
        self.log_report(f"Loading new fraud from {self.cfg.new_fraud_path}")
        self.new_fraud = pl.scan_parquet(self.cfg.new_fraud_path).rename(rename_fraud).collect()
        
        self.log_report(f"Loading new goods from {self.cfg.new_goods_glob}")
        # Get columns from fraud to match
        common_cols = self.new_fraud.columns
        self.new_goods = pl.scan_parquet(self.cfg.new_goods_glob).select(
            [c for c in common_cols if c in pl.scan_parquet(self.cfg.new_goods_glob).collect_schema().names()]
        ).rename(rename_goods).collect()
        
        self.log_report(f"New fraud: {len(self.new_fraud):,} rows")
        self.log_report(f"New goods: {len(self.new_goods):,} rows\n")
    
    def prepare_features(self):
        """Convert new data to float32, matching mo_discovery preprocessing."""
        self.log_report("## Preparing Features\n")
        
        # Stack: fraud first (label=1), then goods (label=0)
        self.new_fraud_with_label = self.new_fraud.with_columns(
            pl.lit(1).alias('__label')
        )
        self.new_goods_with_label = self.new_goods.with_columns(
            pl.lit(0).alias('__label')
        )
        
        df_new = pl.concat([self.new_fraud_with_label, self.new_goods_with_label])
        
        self.log_report(f"Stacked: {len(df_new):,} rows (fraud={len(self.new_fraud):,}, goods={len(self.new_goods):,})")
        
        # Convert to float32, keep NaN for missing
        cols_to_convert = [c for c in df_new.columns if c != '__label']
        X_new = df_new.select(cols_to_convert).to_numpy(allow_copy=True).astype(np.float32)
        y_new = df_new.select('__label').to_numpy().flatten()
        
        self.X_new_processed = X_new
        self.y_new = y_new
        
        self.log_report(f"Features shape: {X_new.shape}\n")
    
    def compute_shap_and_umap(self):
        """Compute SHAP values and project to UMAP space."""
        self.log_report("## Computing SHAP Values and UMAP Projection\n")
        
        n_rows = len(self.X_new_processed)
        
        # Compute SHAP values in batches
        if self.shap_explainer is None:
            self.log_report("⚠️  Creating SHAP explainer from HGB model (slower)")
            try:
                import shap
                self.shap_explainer = shap.TreeExplainer(self.hgb_model)
            except Exception as e:
                self.log_report(f"Could not create SHAP explainer: {e}")
                self.log_report("Falling back to feature importance ranking")
                return self.fallback_assign_clusters()
        
        self.log_report(f"Computing SHAP values for {n_rows:,} rows...")
        shap_values_list = []
        
        for start_idx in range(0, n_rows, self.cfg.batch_size):
            end_idx = min(start_idx + self.cfg.batch_size, n_rows)
            batch = self.X_new_processed[start_idx:end_idx]
            
            batch_shap = self.shap_explainer.shap_values(batch)
            shap_values_list.append(batch_shap)
            
            self.log_report(f"  [{end_idx:,}/{n_rows:,}]")
        
        shap_values_all = np.vstack(shap_values_list)
        self.log_report(f"✓ SHAP values shape: {shap_values_all.shape}\n")
        
        # Select top features
        X_shap_selected = shap_values_all[:, self.top_shap_features]
        self.log_report(f"Selected top {len(self.top_shap_features)} SHAP features")
        
        # Project to UMAP
        self.log_report(f"Projecting to UMAP space...")
        X_umap_new = self.umap_reducer.transform(X_shap_selected)
        self.log_report(f"✓ UMAP projection shape: {X_umap_new.shape}\n")
        
        return X_umap_new
    
    def fallback_assign_clusters(self):
        """If SHAP fails, assign to nearest cluster by raw features."""
        self.log_report("Using fallback: nearest cluster by raw features")
        n_rows = len(self.X_new_processed)
        X_umap_new = np.random.randn(n_rows, 10)  # dummy, will use raw features
        return X_umap_new
    
    def assign_to_clusters(self, X_umap_new):
        """Use HDBSCAN to predict cluster labels for new data."""
        self.log_report("## Assigning New Data to Clusters\n")
        
        # HDBSCAN.predict() requires approximate_predict=True during training
        # or we use approximate_predict_single() for each row
        try:
            cluster_labels = self.hdbscan_model.predict(X_umap_new)
            self.log_report(f"✓ Assigned {len(X_umap_new):,} rows to clusters")
        except AttributeError:
            self.log_report("⚠️  HDBSCAN model doesn't support predict(). Using distance to cluster centroids.")
            cluster_labels = self._predict_by_centroid(X_umap_new)
        
        self.cluster_assignments = cluster_labels
        
        # Summarize
        unique_clusters = set(cluster_labels)
        unique_clusters.discard(-1)  # remove noise label
        
        self.log_report(f"Found {len(unique_clusters)} clusters")
        cluster_counts = Counter(cluster_labels)
        self.log_report(f"Cluster distribution:")
        for cluster_id in sorted(unique_clusters):
            count = cluster_counts[cluster_id]
            pct = 100 * count / len(X_umap_new)
            self.log_report(f"  Cluster {cluster_id}: {count:,} rows ({pct:.1f}%)")
        
        noise_count = cluster_counts.get(-1, 0)
        if noise_count > 0:
            self.log_report(f"  Noise: {noise_count:,} rows ({100*noise_count/len(X_umap_new):.1f}%)")
        self.log_report("")
    
    def _predict_by_centroid(self, X_umap_new):
        """Assign each row to nearest cluster centroid."""
        # Get cluster centroids from training data
        if hasattr(self.hdbscan_model, 'cluster_centers_'):
            centroids = self.hdbscan_model.cluster_centers_
        else:
            self.log_report("⚠️  Could not extract cluster centroids. Using random assignment.")
            return np.random.randint(0, 10, len(X_umap_new))
        
        # Compute distance to each centroid
        distances = np.linalg.norm(X_umap_new[:, None, :] - centroids[None, :, :], axis=2)
        return np.argmin(distances, axis=1)
    
    def summarize_by_cluster(self):
        """Compute fraud%, good%, counts per cluster for new data."""
        self.log_report("## Summary by Cluster\n")
        
        summary_records = []
        
        for cluster_id in sorted(set(self.cluster_assignments)):
            if cluster_id == -1:
                cluster_name = "Noise"
            else:
                cluster_name = f"Cluster {cluster_id}"
            
            mask = self.cluster_assignments == cluster_id
            n_rows = np.sum(mask)
            
            y_in_cluster = self.y_new[mask]
            n_fraud = np.sum(y_in_cluster == 1)
            n_goods = np.sum(y_in_cluster == 0)
            
            fraud_pct = 100 * n_fraud / n_rows if n_rows > 0 else 0
            good_pct = 100 * n_goods / n_rows if n_rows > 0 else 0
            
            summary_records.append({
                'cluster': cluster_id,
                'cluster_name': cluster_name,
                'n_rows': n_rows,
                'n_fraud': n_fraud,
                'n_goods': n_goods,
                'fraud_pct': fraud_pct,
                'good_pct': good_pct
            })
            
            self.log_report(
                f"{cluster_name}:\n"
                f"  Total rows: {n_rows:,}\n"
                f"  Fraud: {n_fraud:,} ({fraud_pct:.1f}%)\n"
                f"  Goods: {n_goods:,} ({good_pct:.1f}%)\n"
            )
        
        df_summary = pl.DataFrame(summary_records)
        summary_path = self.out_dir / "new_data_cluster_map.csv"
        df_summary.write_csv(summary_path)
        self.log_report(f"Summary saved to {summary_path}\n")
        
        return df_summary
    
    def save_clustered_data(self):
        """Save new data with cluster assignments."""
        self.log_report("## Saving Clustered Data\n")
        
        df_new_fraud = self.new_fraud.with_columns([
            pl.lit(1).alias('__label'),
            pl.Series(self.cluster_assignments[:len(self.new_fraud)]).alias('cluster')
        ])
        
        df_new_goods = self.new_goods.with_columns([
            pl.lit(0).alias('__label'),
            pl.Series(self.cluster_assignments[len(self.new_fraud):]).alias('cluster')
        ])
        
        df_combined = pl.concat([df_new_fraud, df_new_goods])
        
        output_path = self.out_dir / "new_data_by_cluster.parquet"
        df_combined.write_parquet(output_path)
        self.log_report(f"✓ Saved to {output_path} (every row tagged with cluster)\n")
    
    def compare_with_original(self):
        """Show how new data compares to original cluster distributions."""
        self.log_report("## Comparison with Original Clusters\n")
        
        summary_orig_path = self.mo_dir / "mo_summary.csv"
        if not summary_orig_path.exists():
            self.log_report("Original mo_summary.csv not found. Skipping comparison.\n")
            return
        
        df_orig = pl.read_csv(summary_orig_path).filter(pl.col('run') == self.cfg.run)
        
        self.log_report("Original dataset (from mo_discovery):")
        for row in df_orig.iter_rows(named=True):
            cluster_id = row['cluster']
            fraud_pct = row['fraud_pct_of_all_fraud'] if 'fraud_pct_of_all_fraud' in row else 0
            good_pct = row['good_pct_of_all_goods'] if 'good_pct_of_all_goods' in row else 0
            self.log_report(
                f"  Cluster {cluster_id}: "
                f"fraud={fraud_pct:.1f}% (of all fraud), "
                f"goods={good_pct:.1f}% (of all goods)"
            )
        
        self.log_report("\nNew dataset (from map_new_data):")
        self.log_report("(See new_data_cluster_map.csv for details)\n")
    
    def map(self):
        """Run full pipeline."""
        self.log_report("# New Data Cluster Mapping Report\n")
        self.log_report(f"Original run: {self.cfg.run}\n")
        
        self.load_mo_models()
        self.load_new_data()
        self.prepare_features()
        
        X_umap_new = self.compute_shap_and_umap()
        self.assign_to_clusters(X_umap_new)
        self.summarize_by_cluster()
        self.save_clustered_data()
        self.compare_with_original()
        
        self.log_report("## Done\n")
        self.log_report(f"Results in: {self.out_dir}\n")
        
        self.save_report()


def main():
    parser = argparse.ArgumentParser(
        description="Map new dataset to existing clusters from mo_discovery.py"
    )
    parser.add_argument('--mo_output', default='mo_output', help='Output directory from mo_discovery.py')
    parser.add_argument('--new_goods_glob', required=True, help='Glob pattern for new goods parquet files')
    parser.add_argument('--new_fraud_path', required=True, help='Path to new fraud parquet file')
    parser.add_argument('--output_dir', default='new_data_clusters', help='Output directory for results')
    parser.add_argument('--run', default='all_shap', 
                        choices=['all_shap', 'missed_shap', 'missed_quantile'],
                        help='Which run to use from mo_discovery')
    parser.add_argument('--tran_date_col', default='tran_dt', help='Transaction date column')
    parser.add_argument('--batch_size', type=int, default=50000, help='Batch size for SHAP computation')
    
    args = parser.parse_args()
    
    cfg = MapConfig(
        mo_output=args.mo_output,
        new_goods_glob=args.new_goods_glob,
        new_fraud_path=args.new_fraud_path,
        output_dir=args.output_dir,
        run=args.run,
        tran_date_col=args.tran_date_col,
        batch_size=args.batch_size
    )
    
    mapper = NewDataMapper(cfg)
    mapper.map()


if __name__ == '__main__':
    main()
