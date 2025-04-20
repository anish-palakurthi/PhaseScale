    def merge_metrics(self):
        """Merge system and detailed metrics on timestamp"""
        system_df = self.load_system_metrics()
        detailed_df = self.load_detailed_metrics()
        
        if system_df.empty or detailed_df.empty:
            logger.warning("Cannot merge metrics: One or both DataFrames are empty")
            return system_df if not system_df.empty else detailed_df
        
        # Merge on timestamp_epoch (as it's more precise than string timestamp)
        # Use outer join to keep all data points
        merged_df = pd.merge(
            system_df, 
            detailed_df,
            on="timestamp_epoch",
            how="outer",
            suffixes=('_sys', '_det')
        )
        
        # Clean up duplicate columns
        if "timestamp_sys" in merged_df.columns and "timestamp_det" in merged_df.columns:
            merged_df["timestamp"] = merged_df["timestamp_sys"].combine_first(merged_df["timestamp_det"])
            merged_df.drop(["timestamp_sys", "timestamp_det"], axis=1, inplace=True)
        
        return merged_df
    
    def calculate_derived_metrics(self, df):
        """Calculate additional metrics derived from raw data"""
        if df.empty:
            return df
        
        # Sort by timestamp
        df = df.sort_values("timestamp_epoch")
        
        # Calculate power consumption changes (energy used)
        power_cols = [col for col in df.columns if col.startswith("rapl_")]
        for col in power_cols:
            df[f"{col}_delta"] = df[col].diff()
        
        # Calculate CPU frequency changes
        freq_cols = [col for col in df.columns if col.endswith("_freq")]
        for col in freq_cols:
            df[f"{col}_delta"] = df[col].diff()
            df[f"{col}_changes"] = (df[col].diff() != 0).astype(int)
        
        # Calculate time deltas
        df["time_delta"] = df["timestamp_epoch"].diff()
        
        # Calculate power per second (where possible)
        for col in power_cols:
            delta_col = f"{col}_delta"
            if delta_col in df.columns:
                df[f"{col}_per_sec"] = df[delta_col] / df["time_delta"]
        
        return df
    
    def process_all(self, output_prefix="processed_metrics"):
        """Process all metrics and save to CSV files"""
        # Process system metrics
        system_df = self.load_system_metrics()
        if not system_df.empty:
            system_output = os.path.join(self.output_dir, f"{output_prefix}_system.csv")
            system_df.to_csv(system_output, index=False)
            logger.info(f"Saved system metrics to {system_output}")
        
        # Process detailed metrics
        detailed_df = self.load_detailed_metrics()
        if not detailed_df.empty:
            detailed_output = os.path.join(self.output_dir, f"{output_prefix}_detailed.csv")
            detailed_df.to_csv(detailed_output, index=False)
            logger.info(f"Saved detailed metrics to {detailed_output}")
        
        # Process merged metrics
        merged_df = self.merge_metrics()
        if not merged_df.empty:
            # Calculate derived metrics
            merged_df = self.calculate_derived_metrics(merged_df)
            
            merged_output = os.path.join(self.output_dir, f"{output_prefix}_merged.csv")
            merged_df.to_csv(merged_output, index=False)
            logger.info(f"Saved merged metrics to {merged_output}")
        
        # Extract metrics for each VM
        if not detailed_df.empty and "vm_names" in detailed_df.columns:
            # Get unique VM names
            all_vms = set()
            for vm_list in detailed_df["vm_names"].dropna():
                all_vms.update(vm_list.split(","))
            
            # Process each VM
            for vm in all_vms:
                if vm:  # Skip empty strings
                    vm_df = self.extract_vm_metrics(vm)
                    if not vm_df.empty:
                        vm_output = os.path.join(self.output_dir, f"{output_prefix}_vm_{vm}.csv")
                        vm_df.to_csv(vm_output, index=False)
                        logger.info(f"Saved metrics for VM {vm} to {vm_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process collected system metrics")
    parser.add_argument("--metrics-dir", required=True, help="Directory containing metrics")
    parser.add_argument("--output-dir", default="./processed_data", help="Directory to store processed data")
    parser.add_argument("--output-prefix", default="processed_metrics", help="Prefix for output files")
    
    args = parser.parse_args()
    
    processor = MetricsProcessor(args.metrics_dir, args.output_dir)
    processor.process_all(args.output_prefix)
