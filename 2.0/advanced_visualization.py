import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

np.random.seed(42)

class AdvancedVisualizer:
    def __init__(self, data_path: str, output_dir: str = '2.0'):
        self.data_path = data_path
        self.output_dir = output_dir
        self.df = None
        self.load_data()
    
    def load_data(self):
        self.df = pd.read_csv(self.data_path)
        print(f"数据加载完成: {len(self.df)} 位参赛者")
    
    def plot_regime_comparison(self):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        regimes = ['Rank_Original', 'Percentage', 'Rank_JudgesSave']
        regime_colors = {'Rank_Original': '#1f77b4', 'Percentage': '#ff7f0e', 'Rank_JudgesSave': '#2ca02c'}
        
        for i, regime in enumerate(regimes):
            regime_data = self.df[self.df['Regime'] == regime]
            
            if len(regime_data) == 0:
                continue
            
            ax = axes[i // 2, i % 2]
            
            survival_weeks = regime_data['Survival Weeks'].values
            tech_ranks = regime_data['Technical Rank'].values
            
            ax.scatter(tech_ranks, survival_weeks, alpha=0.6, 
                      color=regime_colors[regime], s=50, edgecolors='black', linewidth=0.5)
            
            z = np.polyfit(tech_ranks, survival_weeks, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(tech_ranks), max(tech_ranks), 100)
            ax.plot(x_line, p(x_line), color=regime_colors[regime], 
                   linewidth=2, linestyle='--', alpha=0.8)
            
            correlation = np.corrcoef(tech_ranks, survival_weeks)[0, 1]
            
            ax.set_xlabel('Technical Rank', fontsize=11, fontweight='bold')
            ax.set_ylabel('Survival Weeks', fontsize=11, fontweight='bold')
            ax.set_title(f'{regime} (r={correlation:.3f})', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.invert_xaxis()
        
        fig.suptitle('Regime Comparison: Technical Rank vs Survival Weeks', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = f'{self.output_dir}/regime_comparison.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"已保存: {filename}")
        plt.close()
    
    def plot_industry_analysis(self):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        industries = self.df['Industry'].value_counts().head(8).index
        
        for i, industry in enumerate(industries[:4]):
            ax = axes[i // 2, i % 2]
            
            industry_data = self.df[self.df['Industry'] == industry]
            
            for regime in ['Rank_Original', 'Percentage', 'Rank_JudgesSave']:
                regime_data = industry_data[industry_data['Regime'] == regime]
                
                if len(regime_data) > 0:
                    survival_weeks = regime_data['Survival Weeks'].values
                    ax.hist(survival_weeks, bins=12, alpha=0.5, 
                           label=regime, edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel('Survival Weeks', fontsize=11, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
            ax.set_title(f'Industry: {industry}', fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
        
        fig.suptitle('Industry Analysis: Survival Distribution by Regime', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = f'{self.output_dir}/industry_analysis.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"已保存: {filename}")
        plt.close()
    
    def plot_jfg_distribution(self):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        regimes = ['Rank_Original', 'Percentage', 'Rank_JudgesSave']
        
        for i, regime in enumerate(regimes):
            ax = axes[i // 2, i % 2]
            
            regime_data = self.df[self.df['Regime'] == regime]
            
            if len(regime_data) == 0:
                continue
            
            jfg_values = regime_data['JFG'].values
            
            ax.hist(jfg_values, bins=30, alpha=0.7, color='steelblue', 
                   edgecolor='black', linewidth=0.5)
            
            ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
            
            mean_jfg = np.mean(jfg_values)
            median_jfg = np.median(jfg_values)
            
            ax.axvline(mean_jfg, color='green', linestyle=':', linewidth=2, 
                      alpha=0.7, label=f'Mean: {mean_jfg:.3f}')
            ax.axvline(median_jfg, color='orange', linestyle=':', linewidth=2, 
                      alpha=0.7, label=f'Median: {median_jfg:.3f}')
            
            ax.set_xlabel('JFG Index', fontsize=11, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
            ax.set_title(f'{regime} JFG Distribution', fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
        
        fig.delaxes(axes[1, 1])
        fig.suptitle('JFG Index Distribution by Regime', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = f'{self.output_dir}/jfg_distribution.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"已保存: {filename}")
        plt.close()
    
    def plot_survival_premium(self):
        fig, ax = plt.subplots(figsize=(12, 8))
        
        regimes = ['Rank_Original', 'Percentage', 'Rank_JudgesSave']
        regime_colors = {'Rank_Original': '#1f77b4', 'Percentage': '#ff7f0e', 'Rank_JudgesSave': '#2ca02c'}
        
        box_data = []
        labels = []
        
        for regime in regimes:
            regime_data = self.df[self.df['Regime'] == regime]
            
            if len(regime_data) == 0:
                continue
            
            survival_premium = regime_data['Survival Premium'].values
            box_data.append(survival_premium)
            labels.append(regime)
        
        bp = ax.boxplot(box_data, labels=labels, patch_artist=True, 
                      showmeans=True, meanline=True,
                      medianprops=dict(linewidth=2, color='black'),
                      meanprops=dict(linewidth=2, color='red', linestyle='--'),
                      boxprops=dict(linewidth=1.5, alpha=0.7),
                      whiskerprops=dict(linewidth=1.5),
                      capprops=dict(linewidth=1.5))
        
        for patch, regime in zip(bp['boxes'], regimes):
            patch.set_facecolor(regime_colors[regime])
        
        ax.axhline(y=1.0, color='green', linestyle='--', linewidth=2, 
                  alpha=0.7, label='Expected Survival Premium = 1.0')
        
        ax.set_xlabel('Regime', fontsize=12, fontweight='bold')
        ax.set_ylabel('Survival Premium', fontsize=12, fontweight='bold')
        ax.set_title('Survival Premium by Regime', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        filename = f'{self.output_dir}/survival_premium.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"已保存: {filename}")
        plt.close()
    
    def plot_technical_ability_distribution(self):
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        
        quintiles = ['Top20%', '20-40%', '40-60%', '60-80%', 'Bottom20%']
        quintile_colors = {'Top20%': '#2ca02c', '20-40%': '#1f77b4', '40-60%': '#ff7f0e', 
                         '60-80%': '#d62728', 'Bottom20%': '#9467bd'}
        
        for i, quintile in enumerate(quintiles):
            ax = axes[i // 2, i % 2]
            
            quintile_data = self.df[self.df['Technical Quintile'] == quintile]
            
            for regime in ['Rank_Original', 'Percentage', 'Rank_JudgesSave']:
                regime_data = quintile_data[quintile_data['Regime'] == regime]
                
                if len(regime_data) > 0:
                    survival_weeks = regime_data['Survival Weeks'].values
                    ax.hist(survival_weeks, bins=12, alpha=0.5, 
                           label=regime, edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel('Survival Weeks', fontsize=11, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
            ax.set_title(f'Technical Quintile: {quintile}', fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
        
        fig.delaxes(axes[2, 1])
        fig.suptitle('Survival Distribution by Technical Ability Quintile', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = f'{self.output_dir}/technical_ability_distribution.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"已保存: {filename}")
        plt.close()
    
    def plot_placement_analysis(self):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        top_placements = [1, 2, 3, 4]
        
        for i, placement in enumerate(top_placements):
            ax = axes[i // 2, i % 2]
            
            placement_data = self.df[self.df['Placement'] == placement]
            
            for regime in ['Rank_Original', 'Percentage', 'Rank_JudgesSave']:
                regime_data = placement_data[placement_data['Regime'] == regime]
                
                if len(regime_data) > 0:
                    tech_ranks = regime_data['Technical Rank'].values
                    ax.hist(tech_ranks, bins=10, alpha=0.5, 
                           label=regime, edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel('Technical Rank', fontsize=11, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
            ax.set_title(f'Placement: {placement}{"st" if placement == 1 else "nd" if placement == 2 else "rd" if placement == 3 else "th"}', 
                        fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
            ax.invert_xaxis()
        
        fig.suptitle('Technical Rank Distribution by Final Placement', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = f'{self.output_dir}/placement_analysis.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"已保存: {filename}")
        plt.close()
    
    def plot_heterogeneity_heatmap(self):
        fig, ax = plt.subplots(figsize=(14, 10))
        
        industries = self.df['Industry'].value_counts().head(8).index
        quintiles = ['Top20%', '20-40%', '40-60%', '60-80%', 'Bottom20%']
        
        heatmap_data = []
        
        for industry in industries:
            row_data = []
            industry_df = self.df[self.df['Industry'] == industry]
            
            for quintile in quintiles:
                quintile_df = industry_df[industry_df['Technical Quintile'] == quintile]
                
                if len(quintile_df) > 0:
                    avg_survival = quintile_df['Survival Weeks'].mean()
                    row_data.append(avg_survival)
                else:
                    row_data.append(np.nan)
            
            heatmap_data.append(row_data)
        
        heatmap_df = pd.DataFrame(heatmap_data, index=industries, columns=quintiles)
        
        sns.heatmap(heatmap_df, annot=True, fmt='.1f', cmap='YlOrRd', 
                  cbar_kws={'label': 'Average Survival Weeks'}, ax=ax)
        
        ax.set_xlabel('Technical Quintile', fontsize=12, fontweight='bold')
        ax.set_ylabel('Industry', fontsize=12, fontweight='bold')
        ax.set_title('Heterogeneity Heatmap: Survival by Industry and Technical Ability', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        filename = f'{self.output_dir}/heterogeneity_heatmap.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"已保存: {filename}")
        plt.close()
    
    def plot_time_trend(self):
        fig, ax = plt.subplots(figsize=(14, 8))
        
        seasons = sorted(self.df['Season'].unique())
        
        avg_survival = []
        avg_tech_rank = []
        avg_jfg = []
        
        for season in seasons:
            season_data = self.df[self.df['Season'] == season]
            avg_survival.append(season_data['Survival Weeks'].mean())
            avg_tech_rank.append(season_data['Technical Rank'].mean())
            avg_jfg.append(season_data['JFG'].mean())
        
        ax2 = ax.twinx()
        
        line1 = ax.plot(seasons, avg_survival, 'o-', linewidth=2, markersize=8, 
                      color='#1f77b4', label='Avg Survival Weeks')
        line2 = ax2.plot(seasons, avg_tech_rank, 's-', linewidth=2, markersize=8, 
                       color='#ff7f0e', label='Avg Technical Rank')
        line3 = ax2.plot(seasons, avg_jfg, '^-', linewidth=2, markersize=8, 
                       color='#2ca02c', label='Avg JFG')
        
        ax.set_xlabel('Season', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Survival Weeks', fontsize=12, fontweight='bold', 
                    color='#1f77b4')
        ax2.set_ylabel('Technical Rank / JFG', fontsize=12, fontweight='bold')
        ax2.invert_yaxis()
        
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='best', fontsize=10)
        
        ax.grid(True, alpha=0.3)
        
        ax.tick_params(axis='y', labelcolor='#1f77b4')
        
        plt.title('Time Trend: Average Metrics by Season', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = f'{self.output_dir}/time_trend.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"已保存: {filename}")
        plt.close()
    
    def generate_summary_statistics(self):
        summary = []
        summary.append("=" * 80)
        summary.append("2026 MCM Problem C - 制度评估与反事实分析：统计摘要")
        summary.append("=" * 80)
        summary.append("")
        
        summary.append("1. 数据概况")
        summary.append("-" * 80)
        summary.append(f"总参赛人数: {len(self.df)}")
        summary.append(f"赛季范围: {self.df['Season'].min()} - {self.df['Season'].max()}")
        summary.append(f"行业数量: {self.df['Industry'].nunique()}")
        summary.append(f"制度类型: {self.df['Regime'].unique().tolist()}")
        summary.append("")
        
        summary.append("2. 制度比较")
        summary.append("-" * 80)
        for regime in ['Rank_Original', 'Percentage', 'Rank_JudgesSave']:
            regime_data = self.df[self.df['Regime'] == regime]
            if len(regime_data) > 0:
                summary.append(f"\n{regime}:")
                summary.append(f"  参赛人数: {len(regime_data)}")
                summary.append(f"  平均生存周数: {regime_data['Survival Weeks'].mean():.2f}")
                summary.append(f"  平均技术排名: {regime_data['Technical Rank'].mean():.2f}")
                summary.append(f"  平均JFG指数: {regime_data['JFG'].mean():.4f}")
                summary.append(f"  平均生存溢价: {regime_data['Survival Premium'].mean():.4f}")
        
        summary.append("\n3. 行业分析")
        summary.append("-" * 80)
        industry_stats = self.df.groupby('Industry').agg({
            'Survival Weeks': ['mean', 'std'],
            'Technical Rank': ['mean', 'std'],
            'JFG': ['mean', 'std']
        }).round(2)
        
        for industry in self.df['Industry'].value_counts().head(10).index:
            industry_data = self.df[self.df['Industry'] == industry]
            summary.append(f"\n{industry}:")
            summary.append(f"  参赛人数: {len(industry_data)}")
            summary.append(f"  平均生存周数: {industry_data['Survival Weeks'].mean():.2f} ± {industry_data['Survival Weeks'].std():.2f}")
            summary.append(f"  平均技术排名: {industry_data['Technical Rank'].mean():.2f} ± {industry_data['Technical Rank'].std():.2f}")
            summary.append(f"  平均JFG指数: {industry_data['JFG'].mean():.4f} ± {industry_data['JFG'].std():.4f}")
        
        summary.append("\n4. 技术能力分析")
        summary.append("-" * 80)
        for quintile in ['Top20%', '20-40%', '40-60%', '60-80%', 'Bottom20%']:
            quintile_data = self.df[self.df['Technical Quintile'] == quintile]
            if len(quintile_data) > 0:
                summary.append(f"\n{quintile}:")
                summary.append(f"  参赛人数: {len(quintile_data)}")
                summary.append(f"  平均生存周数: {quintile_data['Survival Weeks'].mean():.2f}")
                summary.append(f"  平均最终名次: {quintile_data['Placement'].mean():.2f}")
                summary.append(f"  平均JFG指数: {quintile_data['JFG'].mean():.4f}")
        
        summary.append("\n" + "=" * 80)
        summary.append("统计摘要生成完成")
        summary.append("=" * 80)
        
        summary_text = "\n".join(summary)
        
        with open(f'{self.output_dir}/summary_statistics.txt', 'w', 
                  encoding='utf-8') as f:
            f.write(summary_text)
        
        print(f"已保存: {self.output_dir}/summary_statistics.txt")
        
        return summary_text
    
    def run_all_visualizations(self):
        print("\n开始生成高级可视化图表...")
        
        print("\n1. 制度比较分析")
        self.plot_regime_comparison()
        
        print("\n2. 行业分析")
        self.plot_industry_analysis()
        
        print("\n3. JFG分布分析")
        self.plot_jfg_distribution()
        
        print("\n4. 生存溢价分析")
        self.plot_survival_premium()
        
        print("\n5. 技术能力分布分析")
        self.plot_technical_ability_distribution()
        
        print("\n6. 最终名次分析")
        self.plot_placement_analysis()
        
        print("\n7. 异质性热力图")
        self.plot_heterogeneity_heatmap()
        
        print("\n8. 时间趋势分析")
        self.plot_time_trend()
        
        print("\n9. 生成统计摘要")
        self.generate_summary_statistics()
        
        print("\n所有可视化图表生成完成！")

def main():
    visualizer = AdvancedVisualizer(
        data_path='output/表格11_完整数据集.csv',
        output_dir='2.0'
    )
    
    visualizer.run_all_visualizations()
    
    return visualizer

if __name__ == "__main__":
    main()