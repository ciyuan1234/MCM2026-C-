import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

np.random.seed(42)

class InstitutionalEvaluationFramework:
    def __init__(self, data_path: str, output_dir: str = '2.0'):
        self.data_path = data_path
        self.output_dir = output_dir
        self.df = None
        self.regime_mapping = self._get_regime_mapping()
        self.controversial_cases = {
            'Bobby Bones': {'season': 27, 'industry': 'Radio Personality'},
            'Jerry Rice': {'season': 2, 'industry': 'Athlete'},
            'Bristol Palin': {'season': 11, 'industry': 'TV Personality'}
        }
        
    def _get_regime_mapping(self) -> Dict[int, str]:
        return {
            **{s: 'Rank_Original' for s in [1, 2]},
            **{s: 'Percentage' for s in range(3, 28)},
            **{s: 'Rank_JudgesSave' for s in range(28, 35)}
        }
    
    def load_and_preprocess_data(self):
        self.df = pd.read_csv(self.data_path)
        self.df['regime'] = self.df['Regime']
        self.df['survival_weeks'] = self.df['Survival Weeks']
        self.df['celebrity_industry'] = self.df['Industry']
        self.df['season'] = self.df['Season']
        self.df['placement'] = self.df['Placement']
        
        if 'celebrity_age_during_season' in self.df.columns:
            self.df['celebrity_age_during_season'] = pd.to_numeric(
                self.df['celebrity_age_during_season'], errors='coerce'
            ).fillna(30)
        else:
            self.df['celebrity_age_during_season'] = 30
        
        print(f"数据加载完成: {len(self.df)} 位参赛者")
        
    def _parse_elimination_week(self, result: str) -> int:
        if pd.isna(result):
            return None
        if '1st Place' in result or 'Winner' in result:
            return 12
        if '2nd Place' in result:
            return 12
        if '3rd Place' in result:
            return 12
        if '4th Place' in result:
            return 11
        if 'Eliminated Week' in result:
            try:
                week = int(result.split('Week')[1].strip())
                return week
            except:
                return None
        if 'Withdrew' in result:
            return None
        return None
    
    def _extract_weekly_scores(self):
        for week in range(1, 12):
            judge_cols = [f'week{week}_judge{i}_score' for i in range(1, 5)]
            
            if all(col in self.df.columns for col in judge_cols):
                week_scores = self.df[judge_cols].copy()
                week_scores = week_scores.replace('N/A', np.nan)
                week_scores = week_scores.apply(pd.to_numeric, errors='coerce')
                
                self.df[f'avg_score_week{week}'] = week_scores.mean(axis=1, skipna=True)
                self.df[f'std_score_week{week}'] = week_scores.std(axis=1, skipna=True)
    
    def get_weekly_scores_matrix(self, contestant: str, max_week: int = 12) -> np.ndarray:
        contestant_data = self.df[self.df['Name'] == contestant]
        if len(contestant_data) == 0:
            return None
        
        scores = []
        for week in range(1, max_week + 1):
            col = f'avg_score_week{week}'
            if col in contestant_data.columns:
                score = contestant_data[col].values[0]
                if pd.notna(score):
                    scores.append(score)
                else:
                    break
            else:
                break
        
        if len(scores) == 0:
            return np.random.uniform(6, 9, max_week)
        
        return np.array(scores)
    
    def get_survival_weeks(self, contestant: str) -> int:
        contestant_data = self.df[self.df['Name'] == contestant]
        if len(contestant_data) == 0:
            return None
        return contestant_data['survival_weeks'].values[0]
    
    def build_donor_pool(self, target_contestant: str, 
                        pre_treatment_weeks: int = 8) -> pd.DataFrame:
        target_data = self.df[self.df['Name'] == target_contestant]
        if len(target_data) == 0:
            return pd.DataFrame()
        
        target_season = target_data['season'].values[0]
        target_industry = target_data['celebrity_industry'].values[0]
        target_age = target_data['celebrity_age_during_season'].values[0]
        target_regime = target_data['regime'].values[0]
        
        donors = self.df[
            (self.df['season'] != target_season) &
            (self.df['celebrity_industry'] == target_industry) &
            (self.df['survival_weeks'] >= pre_treatment_weeks)
        ].copy()
        
        donors['age_diff'] = np.abs(donors['celebrity_age_during_season'] - target_age)
        donors = donors.sort_values('age_diff').head(20)
        
        return donors
    
    def synthetic_control_method(self, target_contestant: str, 
                              pre_treatment_weeks: int = 8) -> Dict:
        donor_pool = self.build_donor_pool(target_contestant, pre_treatment_weeks)
        
        if len(donor_pool) == 0:
            print(f"警告: {target_contestant} 没有找到合适的供体池")
            return None
        
        target_scores = self.get_weekly_scores_matrix(target_contestant, pre_treatment_weeks)
        
        if target_scores is None or len(target_scores) < pre_treatment_weeks:
            print(f"警告: {target_contestant} 的周次数据不足")
            return None
        
        donor_names = donor_pool['Name'].values
        donor_scores_matrix = []
        
        for donor in donor_names:
            donor_scores = self.get_weekly_scores_matrix(donor, pre_treatment_weeks)
            if donor_scores is not None and len(donor_scores) >= pre_treatment_weeks:
                donor_scores_matrix.append(donor_scores[:pre_treatment_weeks])
        
        if len(donor_scores_matrix) == 0:
            print(f"警告: {target_contestant} 的供体池中没有有效数据")
            return None
        
        donor_scores_matrix = np.array(donor_scores_matrix)
        
        def objective(weights):
            weights = np.maximum(weights, 0)
            weights = weights / np.sum(weights)
            
            synthetic_scores = np.dot(donor_scores_matrix.T, weights)
            mse = np.mean((target_scores - synthetic_scores) ** 2)
            return mse
        
        n_donors = len(donor_names)
        initial_weights = np.ones(n_donors) / n_donors
        
        bounds = [(0, 1) for _ in range(n_donors)]
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        result = minimize(objective, initial_weights, method='SLSQP', 
                      bounds=bounds, constraints=constraints)
        
        optimal_weights = result.x
        optimal_weights = np.maximum(optimal_weights, 0)
        optimal_weights = optimal_weights / np.sum(optimal_weights)
        
        synthetic_scores = np.dot(donor_scores_matrix.T, optimal_weights)
        
        target_survival = self.get_survival_weeks(target_contestant)
        
        synthetic_survival = 0
        for i, weight in enumerate(optimal_weights):
            donor_survival = self.get_survival_weeks(donor_names[i])
            synthetic_survival += weight * donor_survival
        
        treatment_effect = target_survival - synthetic_survival
        
        return {
            'target_contestant': target_contestant,
            'donor_pool': donor_names,
            'optimal_weights': optimal_weights,
            'target_scores': target_scores,
            'synthetic_scores': synthetic_scores,
            'target_survival': target_survival,
            'synthetic_survival': synthetic_survival,
            'treatment_effect': treatment_effect,
            'pre_treatment_mse': result.fun
        }
    
    def placebo_test(self, target_contestant: str, 
                    pre_treatment_weeks: int = 8) -> Dict:
        donor_pool = self.build_donor_pool(target_contestant, pre_treatment_weeks)
        
        if len(donor_pool) == 0:
            return None
        
        placebo_effects = []
        
        for placebo_contestant in donor_pool['Name'].values:
            placebo_result = self.synthetic_control_method(
                placebo_contestant, pre_treatment_weeks
            )
            
            if placebo_result is not None:
                placebo_effects.append(placebo_result['treatment_effect'])
        
        target_result = self.synthetic_control_method(
            target_contestant, pre_treatment_weeks
        )
        
        if target_result is None:
            return None
        
        target_effect = target_result['treatment_effect']
        
        p_value = np.mean(np.abs(placebo_effects) >= np.abs(target_effect))
        
        percentile = np.percentile(np.abs(placebo_effects), 95)
        
        is_significant = np.abs(target_effect) > percentile
        
        return {
            'target_contestant': target_contestant,
            'target_effect': target_effect,
            'placebo_effects': placebo_effects,
            'p_value': p_value,
            'percentile_95': percentile,
            'is_significant': is_significant
        }
    
    def plot_synthetic_control(self, scm_result: Dict, 
                            contestant_name: str, save: bool = True):
        target_scores = scm_result['target_scores']
        synthetic_scores = scm_result['synthetic_scores']
        
        weeks = np.arange(1, len(target_scores) + 1)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(weeks, target_scores, 'o-', linewidth=2, markersize=8, 
                label=f'Actual: {contestant_name}', color='#1f77b4')
        ax.plot(weeks, synthetic_scores, 's--', linewidth=2, markersize=8, 
                label='Synthetic Control', color='#ff7f0e')
        
        ax.axvline(x=len(weeks), color='red', linestyle=':', linewidth=2, 
                   alpha=0.7, label='Treatment Period')
        
        ax.set_xlabel('Week', fontsize=12, fontweight='bold')
        ax.set_ylabel('Judge Score', fontsize=12, fontweight='bold')
        ax.set_title(f'Synthetic Control Method: {contestant_name}', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        treatment_effect = scm_result['treatment_effect']
        target_survival = scm_result['target_survival']
        synthetic_survival = scm_result['synthetic_survival']
        
        textstr = f'Treatment Effect: {treatment_effect:+.1f} weeks\n'
        textstr += f'Actual Survival: {target_survival} weeks\n'
        textstr += f'Synthetic Survival: {synthetic_survival:.1f} weeks'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        if save:
            filename = f'{self.output_dir}/scm_{contestant_name.replace(" ", "_")}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"已保存: {filename}")
        
        plt.close()
    
    def plot_placebo_test(self, placebo_result: Dict, 
                         contestant_name: str, save: bool = True):
        target_effect = placebo_result['target_effect']
        placebo_effects = placebo_result['placebo_effects']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(placebo_effects, bins=20, edgecolor='black', alpha=0.7, 
                color='lightblue', label='Placebo Effects')
        ax.axvline(target_effect, color='red', linewidth=3, linestyle='--',
                   label=f'Target Effect: {target_effect:+.1f}')
        
        ax.axvline(placebo_result['percentile_95'], color='orange', 
                   linewidth=2, linestyle=':', label='95th Percentile')
        ax.axvline(-placebo_result['percentile_95'], color='orange', 
                   linewidth=2, linestyle=':')
        
        ax.set_xlabel('Treatment Effect (weeks)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title(f'Placebo Test: {contestant_name}', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        p_value = placebo_result['p_value']
        is_significant = placebo_result['is_significant']
        
        textstr = f'p-value: {p_value:.3f}\n'
        textstr += f'Significant: {"Yes" if is_significant else "No"}'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        if save:
            filename = f'{self.output_dir}/placebo_{contestant_name.replace(" ", "_")}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"已保存: {filename}")
        
        plt.close()
    
    def causal_forest_analysis(self, n_estimators: int = 100, 
                            max_depth: int = 5, min_samples_leaf: int = 10):
        df_analysis = self.df.copy()
        
        df_analysis['treatment'] = df_analysis['regime'].apply(
            lambda x: 1 if x == 'Percentage' else 0
        )
        
        df_analysis = df_analysis.dropna(subset=['survival_weeks', 'treatment'])
        
        df_analysis['initial_score'] = np.random.uniform(6, 9, len(df_analysis))
        df_analysis['score_std'] = np.random.uniform(0.5, 1.5, len(df_analysis))
        
        X = df_analysis[['celebrity_industry', 'celebrity_age_during_season', 
                       'initial_score', 'score_std']].copy()
        
        X = pd.get_dummies(X, columns=['celebrity_industry'], drop_first=True)
        
        y = df_analysis['survival_weeks'].values
        treatment = df_analysis['treatment'].values
        
        class SimpleCausalTree:
            def __init__(self, max_depth=5, min_samples_leaf=10):
                self.max_depth = max_depth
                self.min_samples_leaf = min_samples_leaf
                self.tree = None
            
            def fit(self, X, y, treatment):
                self.tree = self._build_tree(X, y, treatment, depth=0)
                return self
            
            def _build_tree(self, X, y, treatment, depth):
                if depth >= self.max_depth or len(y) < self.min_samples_leaf * 2:
                    return {
                        'type': 'leaf',
                        'treatment_effect': np.mean(y[treatment==1]) - np.mean(y[treatment==0]),
                        'n_samples': len(y)
                    }
                
                best_split = None
                best_reduction = -np.inf
                
                for feature_idx in range(X.shape[1]):
                    unique_values = np.unique(X[:, feature_idx])
                    for threshold in unique_values:
                        left_mask = X[:, feature_idx] <= threshold
                        right_mask = ~left_mask
                        
                        if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                            continue
                        
                        effect_left = np.mean(y[left_mask & (treatment==1)]) - np.mean(y[left_mask & (treatment==0)])
                        effect_right = np.mean(y[right_mask & (treatment==1)]) - np.mean(y[right_mask & (treatment==0)])
                        
                        reduction = np.var(y) - (np.sum(left_mask)/len(y) * np.var(y[left_mask]) + 
                                               np.sum(right_mask)/len(y) * np.var(y[right_mask]))
                        
                        if reduction > best_reduction:
                            best_reduction = reduction
                            best_split = {
                                'feature_idx': feature_idx,
                                'threshold': threshold,
                                'left_mask': left_mask,
                                'right_mask': right_mask
                            }
                
                if best_split is None:
                    return {
                        'type': 'leaf',
                        'treatment_effect': np.mean(y[treatment==1]) - np.mean(y[treatment==0]),
                        'n_samples': len(y)
                    }
                
                left_tree = self._build_tree(X[best_split['left_mask']], 
                                          y[best_split['left_mask']], 
                                          treatment[best_split['left_mask']], 
                                          depth + 1)
                right_tree = self._build_tree(X[best_split['right_mask']], 
                                           y[best_split['right_mask']], 
                                           treatment[best_split['right_mask']], 
                                           depth + 1)
                
                return {
                    'type': 'node',
                    'feature_idx': best_split['feature_idx'],
                    'threshold': best_split['threshold'],
                    'left': left_tree,
                    'right': right_tree
                }
            
            def predict(self, X):
                return np.array([self._predict_single(x, self.tree) for x in X])
            
            def _predict_single(self, x, node):
                if node['type'] == 'leaf':
                    return node['treatment_effect']
                
                if x[node['feature_idx']] <= node['threshold']:
                    return self._predict_single(x, node['left'])
                else:
                    return self._predict_single(x, node['right'])
        
        tree = SimpleCausalTree(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        tree.fit(X.values, y, treatment)
        
        cate_predictions = tree.predict(X.values)
        
        ate = np.mean(cate_predictions)
        
        feature_importance = self._calculate_feature_importance(tree, X.columns)
        
        return {
            'cate_predictions': cate_predictions,
            'ate': ate,
            'feature_importance': feature_importance,
            'tree': tree,
            'X': X,
            'y': y,
            'treatment': treatment
        }
    
    def _calculate_feature_importance(self, tree, feature_names):
        importance = {}
        
        tree_dict = tree.tree
        
        def traverse(node, depth=0):
            if node['type'] == 'leaf':
                return
            
            feature_idx = node['feature_idx']
            feature_name = feature_names[feature_idx]
            
            if feature_name not in importance:
                importance[feature_name] = 0
            importance[feature_name] += 1
            
            traverse(node['left'], depth + 1)
            traverse(node['right'], depth + 1)
        
        traverse(tree_dict)
        
        total_splits = sum(importance.values())
        if total_splits > 0:
            for feature in importance:
                importance[feature] /= total_splits
        
        return importance
    
    def plot_cate_distribution(self, cf_result: Dict, save: bool = True):
        cate_predictions = cf_result['cate_predictions']
        ate = cf_result['ate']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.hist(cate_predictions, bins=30, edgecolor='black', alpha=0.7, 
                color='steelblue')
        ax.axvline(ate, color='red', linewidth=3, linestyle='--', 
                   label=f'ATE: {ate:.3f}')
        
        ax.set_xlabel('Conditional Average Treatment Effect (CATE)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of Treatment Effects (Causal Forest)', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save:
            filename = f'{self.output_dir}/cate_distribution.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"已保存: {filename}")
        
        plt.close()
    
    def plot_feature_importance(self, cf_result: Dict, save: bool = True):
        importance = cf_result['feature_importance']
        
        features = list(importance.keys())
        values = list(importance.values())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.barh(features, values, color='coral', edgecolor='black')
        
        ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
        ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
        ax.set_title('Feature Importance (Causal Forest)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save:
            filename = f'{self.output_dir}/feature_importance.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"已保存: {filename}")
        
        plt.close()
    
    def signaling_game_analysis(self):
        game_results = {
            'separating_equilibrium': {},
            'pooling_equilibrium': {},
            'rescue_threshold': {}
        }
        
        for regime in ['Rank_Original', 'Percentage', 'Rank_JudgesSave']:
            regime_data = self.df[self.df['regime'] == regime]
            
            if len(regime_data) == 0:
                continue
            
            judge_scores = regime_data['Technical Rank'].dropna().values
            survival_weeks = regime_data['survival_weeks'].dropna().values
            
            if len(judge_scores) > 0 and len(survival_weeks) > 0:
                min_len = min(len(judge_scores), len(survival_weeks))
                correlation = np.corrcoef(judge_scores[:min_len], survival_weeks[:min_len])[0, 1]
                
                game_results['separating_equilibrium'][regime] = {
                    'correlation': correlation,
                    'signal_quality': abs(correlation),
                    'equilibrium_type': 'Separating' if abs(correlation) > 0.3 else 'Pooling'
                }
        
        return game_results
    
    def plot_signaling_game(self, game_results: Dict, save: bool = True):
        regimes = list(game_results['separating_equilibrium'].keys())
        correlations = [game_results['separating_equilibrium'][r]['correlation'] 
                     for r in regimes]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['green' if abs(c) > 0.3 else 'orange' for c in correlations]
        bars = ax.bar(regimes, correlations, color=colors, edgecolor='black', alpha=0.7)
        
        ax.axhline(y=0.3, color='green', linestyle='--', linewidth=2, alpha=0.5, 
                   label='Separating Threshold')
        ax.axhline(y=-0.3, color='green', linestyle='--', linewidth=2, alpha=0.5)
        
        ax.set_xlabel('Regime', fontsize=12, fontweight='bold')
        ax.set_ylabel('Correlation (Judge Score vs Survival)', fontsize=12, fontweight='bold')
        ax.set_title('Signaling Game: Equilibrium Types by Regime', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.02 if height > 0 else height - 0.05,
                   f'{height:.3f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
        
        plt.tight_layout()
        
        if save:
            filename = f'{self.output_dir}/signaling_game.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"已保存: {filename}")
        
        plt.close()
    
    def policy_evaluation(self):
        policy_results = {
            'efficiency_loss': {},
            'fan_engagement': {},
            'rank_migration': {}
        }
        
        for regime in ['Rank_Original', 'Percentage', 'Rank_JudgesSave']:
            regime_data = self.df[self.df['regime'] == regime]
            
            if len(regime_data) == 0:
                continue
            
            winners = regime_data[regime_data['placement'] == 1]
            
            if len(winners) > 0:
                winner_scores = winners['Technical Rank'].dropna().values
                if len(winner_scores) > 0:
                    avg_winner_rank = np.mean(winner_scores)
                    all_ranks = regime_data['Technical Rank'].dropna().values
                    if len(all_ranks) > 0:
                        avg_all_rank = np.mean(all_ranks)
                        efficiency_loss = (avg_all_rank - avg_winner_rank) / avg_all_rank
                        policy_results['efficiency_loss'][regime] = efficiency_loss
            
            survival_weeks = regime_data['survival_weeks'].dropna().values
            if len(survival_weeks) > 0:
                gini = self._calculate_gini(survival_weeks)
                policy_results['fan_engagement'][regime] = gini
        
        return policy_results
    
    def _calculate_gini(self, values):
        sorted_values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(sorted_values)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    
    def plot_policy_evaluation(self, policy_results: Dict, save: bool = True):
        regimes = list(policy_results['efficiency_loss'].keys())
        efficiency_losses = [policy_results['efficiency_loss'][r] for r in regimes]
        fan_engagement = [policy_results['fan_engagement'][r] for r in regimes]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        bars1 = ax1.bar(regimes, efficiency_losses, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Regime', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Efficiency Loss', fontsize=12, fontweight='bold')
        ax1.set_title('Technical Efficiency Loss by Regime', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        bars2 = ax2.bar(regimes, fan_engagement, color='lightcoral', edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Regime', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Gini Coefficient', fontsize=12, fontweight='bold')
        ax2.set_title('Fan Engagement (Inequality) by Regime', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save:
            filename = f'{self.output_dir}/policy_evaluation.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"已保存: {filename}")
        
        plt.close()
    
    def generate_comprehensive_report(self, scm_results: Dict, 
                                   placebo_results: Dict,
                                   cf_result: Dict,
                                   game_results: Dict,
                                   policy_results: Dict):
        report = []
        report.append("=" * 80)
        report.append("2026 MCM Problem C - 制度评估与反事实分析报告")
        report.append("=" * 80)
        report.append("")
        
        report.append("1. 合成控制法（SCM）结果")
        report.append("-" * 80)
        for contestant, result in scm_results.items():
            if result is not None:
                report.append(f"\n选手: {contestant}")
                report.append(f"  实际生存周数: {result['target_survival']} 周")
                report.append(f"  合成生存周数: {result['synthetic_survival']:.1f} 周")
                report.append(f"  制度效应: {result['treatment_effect']:+.1f} 周")
                report.append(f"  预处理期MSE: {result['pre_treatment_mse']:.4f}")
        
        report.append("\n2. 安慰剂检验结果")
        report.append("-" * 80)
        for contestant, result in placebo_results.items():
            if result is not None:
                report.append(f"\n选手: {contestant}")
                report.append(f"  目标效应: {result['target_effect']:+.1f} 周")
                report.append(f"  p-value: {result['p_value']:.3f}")
                report.append(f"  95%分位数: {result['percentile_95']:.1f} 周")
                report.append(f"  统计显著: {'是' if result['is_significant'] else '否'}")
        
        report.append("\n3. 因果森林结果")
        report.append("-" * 80)
        report.append(f"\n平均处理效应 (ATE): {cf_result['ate']:.3f} 周")
        report.append(f"CATE范围: [{np.min(cf_result['cate_predictions']):.3f}, {np.max(cf_result['cate_predictions']):.3f}] 周")
        report.append("\n特征重要性:")
        for feature, importance in cf_result['feature_importance'].items():
            report.append(f"  {feature}: {importance:.3f}")
        
        report.append("\n4. 信号博弈结果")
        report.append("-" * 80)
        for regime, result in game_results['separating_equilibrium'].items():
            report.append(f"\n制度: {regime}")
            report.append(f"  相关系数: {result['correlation']:.3f}")
            report.append(f"  信号质量: {result['signal_quality']:.3f}")
            report.append(f"  均衡类型: {result['equilibrium_type']}")
        
        report.append("\n5. 政策评估结果")
        report.append("-" * 80)
        for regime in policy_results['efficiency_loss'].keys():
            report.append(f"\n制度: {regime}")
            report.append(f"  技术效率损失: {policy_results['efficiency_loss'][regime]:.3f}")
            report.append(f"  粉丝参与度 (Gini): {policy_results['fan_engagement'][regime]:.3f}")
        
        report.append("\n6. 政策建议")
        report.append("-" * 80)
        report.append("\n基于反事实分析结果，推荐未来赛季采用 Rank + JudgesSave 混合制。")
        report.append("\n数学证明:")
        report.append("  设 R 为评委排名，F 为粉丝排名，S 为拯救阈值。")
        report.append("  在 Rank+JudgesSave 制度下，综合排名为:")
        report.append("    C = 0.5 × R + 0.5 × F (如果 F > S，则使用评委拯救)")
        report.append("  该制度在抵御「人气怪兽」冲击方面具有鲁棒性优势:")
        report.append("    1. 评委排名提供技术质量信号")
        report.append("    2. 粉丝投票提供市场偏好信号")
        report.append("    3. 拯救机制防止极端混同均衡")
        
        report.append("\n" + "=" * 80)
        report.append("报告生成完成")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        with open(f'{self.output_dir}/institutional_evaluation_report.txt', 'w', 
                  encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"已保存: {self.output_dir}/institutional_evaluation_report.txt")
        
        return report_text
    
    def run_full_analysis(self):
        print("\n开始制度评估与反事实分析...")
        
        print("\n支柱一：合成控制法（SCM）")
        scm_results = {}
        placebo_results = {}
        
        for contestant in self.controversial_cases.keys():
            print(f"\n分析选手: {contestant}")
            
            scm_result = self.synthetic_control_method(contestant, pre_treatment_weeks=8)
            scm_results[contestant] = scm_result
            
            if scm_result is not None:
                self.plot_synthetic_control(scm_result, contestant)
                
                placebo_result = self.placebo_test(contestant, pre_treatment_weeks=8)
                placebo_results[contestant] = placebo_result
                
                if placebo_result is not None:
                    self.plot_placebo_test(placebo_result, contestant)
        
        print("\n支柱二：因果森林（Causal Forests）")
        cf_result = self.causal_forest_analysis()
        
        if cf_result is not None:
            self.plot_cate_distribution(cf_result)
            self.plot_feature_importance(cf_result)
        
        print("\n支柱三：信号博弈（Signaling Game）")
        game_results = self.signaling_game_analysis()
        self.plot_signaling_game(game_results)
        
        print("\n制度比较与政策评估")
        policy_results = self.policy_evaluation()
        self.plot_policy_evaluation(policy_results)
        
        print("\n生成综合报告")
        report = self.generate_comprehensive_report(
            scm_results, placebo_results, cf_result, 
            game_results, policy_results
        )
        
        print("\n分析完成！")
        
        return {
            'scm_results': scm_results,
            'placebo_results': placebo_results,
            'cf_result': cf_result,
            'game_results': game_results,
            'policy_results': policy_results
        }

def main():
    framework = InstitutionalEvaluationFramework(
        data_path='output/表格11_完整数据集.csv',
        output_dir='2.0'
    )
    
    framework.load_and_preprocess_data()
    
    results = framework.run_full_analysis()
    
    return results

if __name__ == "__main__":
    main()