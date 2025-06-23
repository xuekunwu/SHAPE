"""
Results Visualization for LLM Performance Evaluation
Generate comprehensive visualizations of evaluation results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import logging

# Configure matplotlib for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResultsVisualizer:
    """Class for creating comprehensive visualizations of evaluation results"""
    
    def __init__(self, results_file: str, output_dir: str):
        """
        Initialize the visualizer
        
        Args:
            results_file: Path to evaluation results JSON file
            output_dir: Directory to save visualizations
        """
        self.results_file = Path(results_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load results
        with open(self.results_file, 'r', encoding='utf-8') as f:
            self.results = json.load(f)
        
        # Convert to DataFrame for easier analysis
        self.df = self._prepare_dataframe()
        
        logger.info(f"Loaded results for {len(self.df['model'].unique())} models")
    
    def _prepare_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame for analysis"""
        df_data = []
        
        for model_name, model_results in self.results.get("model_results", {}).items():
            if "test_results" in model_results:
                for test_result in model_results["test_results"]:
                    df_data.append({
                        "model": model_name,
                        "test_case": test_result["test_case_id"],
                        "success": test_result["success"],
                        "execution_time": test_result["execution_time"],
                        "output_quality": test_result["output_quality"],
                        "cost": test_result["cost"],
                        "tokens_used": test_result["tokens_used"],
                        "user_satisfaction": test_result["user_satisfaction"],
                        "error_count": len(test_result["errors"])
                    })
        
        return pd.DataFrame(df_data)
    
    def create_performance_dashboard(self, figsize=(20, 15)):
        """Create comprehensive performance dashboard"""
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        fig.suptitle("LLM Performance Evaluation Dashboard", fontsize=16, fontweight='bold')
        
        # 1. Success Rate by Model
        success_rates = self.df.groupby("model")["success"].mean()
        axes[0, 0].bar(success_rates.index, success_rates.values, color='skyblue', alpha=0.7)
        axes[0, 0].set_title("Success Rate by Model", fontweight='bold')
        axes[0, 0].set_ylabel("Success Rate")
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Average Execution Time
        avg_times = self.df.groupby("model")["execution_time"].mean()
        axes[0, 1].bar(avg_times.index, avg_times.values, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title("Average Execution Time", fontweight='bold')
        axes[0, 1].set_ylabel("Time (seconds)")
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Output Quality Distribution
        sns.boxplot(data=self.df, x="model", y="output_quality", ax=axes[0, 2], palette="Set3")
        axes[0, 2].set_title("Output Quality Distribution", fontweight='bold')
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Total Cost Analysis
        total_costs = self.df.groupby("model")["cost"].sum()
        axes[1, 0].bar(total_costs.index, total_costs.values, color='lightgreen', alpha=0.7)
        axes[1, 0].set_title("Total Cost per Model", fontweight='bold')
        axes[1, 0].set_ylabel("Cost ($)")
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Token Efficiency
        token_efficiency = self.df.groupby("model").apply(
            lambda x: x["tokens_used"].sum() / x["success"].sum() if x["success"].sum() > 0 else float('inf')
        )
        axes[1, 1].bar(token_efficiency.index, token_efficiency.values, color='gold', alpha=0.7)
        axes[1, 1].set_title("Token Efficiency (tokens/success)", fontweight='bold')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. User Satisfaction
        avg_satisfaction = self.df.groupby("model")["user_satisfaction"].mean()
        axes[1, 2].bar(avg_satisfaction.index, avg_satisfaction.values, color='plum', alpha=0.7)
        axes[1, 2].set_title("Average User Satisfaction", fontweight='bold')
        axes[1, 2].set_ylabel("Satisfaction Score (0-5)")
        axes[1, 2].tick_params(axis='x', rotation=45)
        axes[1, 2].grid(True, alpha=0.3)
        
        # 7. Performance vs Cost Scatter
        performance_cost = self.df.groupby("model").agg({
            "output_quality": "mean",
            "cost": "sum"
        })
        scatter = axes[2, 0].scatter(performance_cost["cost"], performance_cost["output_quality"], 
                                   s=100, alpha=0.7, c=range(len(performance_cost)), cmap='viridis')
        for idx, row in performance_cost.iterrows():
            axes[2, 0].annotate(idx, (row["cost"], row["output_quality"]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[2, 0].set_xlabel("Total Cost ($)")
        axes[2, 0].set_ylabel("Average Output Quality")
        axes[2, 0].set_title("Performance vs Cost", fontweight='bold')
        axes[2, 0].grid(True, alpha=0.3)
        
        # 8. Error Analysis
        error_counts = self.df.groupby("model")["error_count"].sum()
        axes[2, 1].bar(error_counts.index, error_counts.values, color='salmon', alpha=0.7)
        axes[2, 1].set_title("Total Errors by Model", fontweight='bold')
        axes[2, 1].set_ylabel("Error Count")
        axes[2, 1].tick_params(axis='x', rotation=45)
        axes[2, 1].grid(True, alpha=0.3)
        
        # 9. Test Case Success Heatmap
        pivot_data = self.df.pivot_table(index="model", columns="test_case", 
                                       values="success", aggfunc="mean")
        sns.heatmap(pivot_data, annot=True, cmap="RdYlGn", center=0.5, 
                   ax=axes[2, 2], cbar_kws={"label": "Success Rate"})
        axes[2, 2].set_title("Test Case Success Heatmap", fontweight='bold')
        axes[2, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "performance_dashboard.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Performance dashboard saved to {self.output_dir / 'performance_dashboard.png'}")
    
    def create_radar_chart(self, figsize=(12, 10)):
        """Create radar chart comparing model performance across multiple dimensions"""
        # Calculate normalized metrics for each model
        metrics = {}
        for model in self.df["model"].unique():
            model_data = self.df[self.df["model"] == model]
            
            # Normalize metrics to 0-1 scale
            success_rate = model_data["success"].mean()
            avg_quality = model_data["output_quality"].mean() / 10.0  # Normalize to 0-1
            avg_satisfaction = model_data["user_satisfaction"].mean() / 5.0  # Normalize to 0-1
            
            # Normalize execution time (lower is better)
            avg_time = model_data["execution_time"].mean()
            normalized_time = max(0, 1 - (avg_time / 60.0))  # Cap at 60 seconds
            
            # Normalize cost (lower is better)
            total_cost = model_data["cost"].sum()
            normalized_cost = max(0, 1 - (total_cost / 1.0))  # Cap at $1.00
            
            # Calculate efficiency (success per token)
            total_tokens = model_data["tokens_used"].sum()
            success_count = model_data["success"].sum()
            efficiency = success_count / total_tokens if total_tokens > 0 else 0
            normalized_efficiency = min(1.0, efficiency * 1000)  # Scale appropriately
            
            metrics[model] = {
                "Success Rate": success_rate,
                "Output Quality": avg_quality,
                "User Satisfaction": avg_satisfaction,
                "Speed": normalized_time,
                "Cost Efficiency": normalized_cost,
                "Token Efficiency": normalized_efficiency
            }
        
        # Create radar chart
        categories = list(metrics[list(metrics.keys())[0]].keys())
        N = len(categories)
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        
        # Plot each model
        colors = plt.cm.Set3(np.linspace(0, 1, len(metrics)))
        for i, (model, values) in enumerate(metrics.items()):
            values_list = list(values.values())
            values_list += values_list[:1]  # Complete the circle
            
            ax.plot(angles, values_list, 'o-', linewidth=2, label=model, color=colors[i])
            ax.fill(angles, values_list, alpha=0.1, color=colors[i])
        
        # Customize the plot
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title("Model Performance Comparison (Radar Chart)", fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "radar_chart.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Radar chart saved to {self.output_dir / 'radar_chart.png'}")
    
    def create_cost_analysis(self, figsize=(15, 10)):
        """Create detailed cost analysis visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle("Cost Analysis Dashboard", fontsize=16, fontweight='bold')
        
        # 1. Cost per successful task
        cost_per_success = self.df.groupby("model").apply(
            lambda x: x["cost"].sum() / x["success"].sum() if x["success"].sum() > 0 else float('inf')
        )
        axes[0, 0].bar(cost_per_success.index, cost_per_success.values, color='lightblue', alpha=0.7)
        axes[0, 0].set_title("Cost per Successful Task", fontweight='bold')
        axes[0, 0].set_ylabel("Cost ($)")
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Cost distribution by test case
        cost_by_test = self.df.groupby(["model", "test_case"])["cost"].sum().reset_index()
        pivot_cost = cost_by_test.pivot(index="model", columns="test_case", values="cost")
        sns.heatmap(pivot_cost, annot=True, fmt=".4f", cmap="YlOrRd", ax=axes[0, 1])
        axes[0, 1].set_title("Cost by Test Case", fontweight='bold')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Cost vs Quality scatter
        cost_quality = self.df.groupby("model").agg({
            "cost": "sum",
            "output_quality": "mean"
        })
        scatter = axes[1, 0].scatter(cost_quality["cost"], cost_quality["output_quality"], 
                                   s=100, alpha=0.7, c=range(len(cost_quality)), cmap='viridis')
        for idx, row in cost_quality.iterrows():
            axes[1, 0].annotate(idx, (row["cost"], row["output_quality"]), 
                              xytext=(5, 5), textcoords='offset points')
        axes[1, 0].set_xlabel("Total Cost ($)")
        axes[1, 0].set_ylabel("Average Output Quality")
        axes[1, 0].set_title("Cost vs Quality", fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Cost efficiency ranking
        efficiency_scores = []
        for model in self.df["model"].unique():
            model_data = self.df[self.df["model"] == model]
            total_cost = model_data["cost"].sum()
            avg_quality = model_data["output_quality"].mean()
            success_rate = model_data["success"].mean()
            
            # Efficiency score: quality * success_rate / cost
            efficiency = (avg_quality * success_rate) / total_cost if total_cost > 0 else 0
            efficiency_scores.append((model, efficiency))
        
        efficiency_scores.sort(key=lambda x: x[1], reverse=True)
        models, scores = zip(*efficiency_scores)
        
        axes[1, 1].bar(models, scores, color='lightgreen', alpha=0.7)
        axes[1, 1].set_title("Cost Efficiency Ranking", fontweight='bold')
        axes[1, 1].set_ylabel("Efficiency Score")
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "cost_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Cost analysis saved to {self.output_dir / 'cost_analysis.png'}")
    
    def create_test_case_analysis(self, figsize=(15, 12)):
        """Create detailed analysis of test case performance"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle("Test Case Performance Analysis", fontsize=16, fontweight='bold')
        
        # 1. Success rate by test case
        test_success = self.df.groupby("test_case")["success"].mean()
        axes[0, 0].bar(range(len(test_success)), test_success.values, color='skyblue', alpha=0.7)
        axes[0, 0].set_title("Success Rate by Test Case", fontweight='bold')
        axes[0, 0].set_ylabel("Success Rate")
        axes[0, 0].set_xticks(range(len(test_success)))
        axes[0, 0].set_xticklabels(test_success.index, rotation=45, ha='right')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Average execution time by test case
        test_time = self.df.groupby("test_case")["execution_time"].mean()
        axes[0, 1].bar(range(len(test_time)), test_time.values, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title("Average Execution Time by Test Case", fontweight='bold')
        axes[0, 1].set_ylabel("Time (seconds)")
        axes[0, 1].set_xticks(range(len(test_time)))
        axes[0, 1].set_xticklabels(test_time.index, rotation=45, ha='right')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Output quality by test case
        test_quality = self.df.groupby("test_case")["output_quality"].mean()
        axes[1, 0].bar(range(len(test_quality)), test_quality.values, color='lightgreen', alpha=0.7)
        axes[1, 0].set_title("Average Output Quality by Test Case", fontweight='bold')
        axes[1, 0].set_ylabel("Quality Score")
        axes[1, 0].set_xticks(range(len(test_quality)))
        axes[1, 0].set_xticklabels(test_quality.index, rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Test case difficulty analysis
        test_difficulty = self.df.groupby("test_case").agg({
            "success": "mean",
            "execution_time": "mean",
            "output_quality": "mean"
        })
        
        # Create difficulty score (lower success + higher time + lower quality = more difficult)
        test_difficulty["difficulty_score"] = (
            (1 - test_difficulty["success"]) * 0.4 +
            (test_difficulty["execution_time"] / 60) * 0.3 +
            ((10 - test_difficulty["output_quality"]) / 10) * 0.3
        )
        
        axes[1, 1].bar(range(len(test_difficulty)), test_difficulty["difficulty_score"].values, 
                      color='gold', alpha=0.7)
        axes[1, 1].set_title("Test Case Difficulty Score", fontweight='bold')
        axes[1, 1].set_ylabel("Difficulty Score")
        axes[1, 1].set_xticks(range(len(test_difficulty)))
        axes[1, 1].set_xticklabels(test_difficulty.index, rotation=45, ha='right')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "test_case_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Test case analysis saved to {self.output_dir / 'test_case_analysis.png'}")
    
    def create_summary_statistics(self):
        """Create summary statistics table"""
        summary_stats = {}
        
        for model in self.df["model"].unique():
            model_data = self.df[self.df["model"] == model]
            
            summary_stats[model] = {
                "Total Tests": len(model_data),
                "Success Rate": f"{model_data['success'].mean():.3f}",
                "Avg Execution Time": f"{model_data['execution_time'].mean():.2f}s",
                "Avg Output Quality": f"{model_data['output_quality'].mean():.2f}/10",
                "Total Cost": f"${model_data['cost'].sum():.4f}",
                "Avg User Satisfaction": f"{model_data['user_satisfaction'].mean():.2f}/5",
                "Total Tokens": f"{model_data['tokens_used'].sum():,}",
                "Error Count": model_data['error_count'].sum()
            }
        
        # Convert to DataFrame for better display
        summary_df = pd.DataFrame(summary_stats).T
        
        # Save as CSV
        summary_file = self.output_dir / "summary_statistics.csv"
        summary_df.to_csv(summary_file)
        
        # Create formatted table
        fig, ax = plt.subplots(figsize=(12, len(summary_stats) * 0.4 + 2))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=summary_df.values,
                        rowLabels=summary_df.index,
                        colLabels=summary_df.columns,
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Color header
        for i in range(len(summary_df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title("Model Performance Summary Statistics", fontsize=16, fontweight='bold', pad=20)
        plt.savefig(self.output_dir / "summary_statistics.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Summary statistics saved to {self.output_dir / 'summary_statistics.csv'}")
        logger.info(f"Summary table saved to {self.output_dir / 'summary_statistics.png'}")
    
    def create_all_visualizations(self):
        """Create all visualizations"""
        logger.info("Creating comprehensive visualizations...")
        
        try:
            self.create_performance_dashboard()
            self.create_radar_chart()
            self.create_cost_analysis()
            self.create_test_case_analysis()
            self.create_summary_statistics()
            
            logger.info("All visualizations completed successfully!")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            raise

def main():
    """Main function to run visualization"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create visualizations for LLM evaluation results")
    parser.add_argument("--results-file", required=True, help="Path to evaluation results JSON file")
    parser.add_argument("--output-dir", required=True, help="Output directory for visualizations")
    parser.add_argument("--dashboard-only", action="store_true", help="Create only the main dashboard")
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = ResultsVisualizer(args.results_file, args.output_dir)
    
    if args.dashboard_only:
        visualizer.create_performance_dashboard()
    else:
        visualizer.create_all_visualizations()

if __name__ == "__main__":
    main() 