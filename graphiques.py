import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import warnings
warnings.filterwarnings('ignore')

# Configuration pour les graphiques
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class WineExplorer:
    def __init__(self, save_dir="wine_plots"):
        """
        Initialise l'explorateur de données Wine
        """
        self.save_dir = save_dir
        self.create_save_directory()
        self.load_data()
        
    def create_save_directory(self):
        """Crée le dossier pour sauvegarder les graphiques"""
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print(f"Dossier '{self.save_dir}' créé pour sauvegarder les graphiques")
        
    def load_data(self):
        """Charge le dataset Wine"""
        # Chargement des données
        wine_data = load_wine()
        
        # Création du DataFrame
        self.df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
        self.df['target'] = wine_data.target
        self.df['class_name'] = wine_data.target_names[wine_data.target]
        
        self.feature_names = wine_data.feature_names
        self.target_names = wine_data.target_names
        
        print("Dataset Wine chargé avec succès!")
        print(f"Forme du dataset: {self.df.shape}")
        print(f"Classes: {self.target_names}")
        
    def basic_info(self):
        """Affiche les informations de base sur le dataset"""
        print("\n" + "="*60)
        print("INFORMATIONS GÉNÉRALES SUR LE DATASET")
        print("="*60)
        
        print(f"Nombre d'échantillons: {len(self.df)}")
        print(f"Nombre de caractéristiques: {len(self.feature_names)}")
        print(f"Nombre de classes: {len(self.target_names)}")
        
        print("\nRépartition des classes:")
        class_counts = self.df['class_name'].value_counts()
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} échantillons ({count/len(self.df)*100:.1f}%)")
        
        print("\nPremières statistiques descriptives:")
        print(self.df[self.feature_names].describe())
        
    def plot_class_distribution(self):
        """Graphique de répartition des classes"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Graphique en barres
        class_counts = self.df['class_name'].value_counts()
        bars = ax1.bar(class_counts.index, class_counts.values, 
                      color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax1.set_title('Répartition des Classes de Vins', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Classe de Vin')
        ax1.set_ylabel('Nombre d\'échantillons')
        
        # Ajout des valeurs sur les barres
        for bar, count in zip(bars, class_counts.values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # Graphique camembert
        ax2.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
               colors=['#FF6B6B', '#4ECDC4', '#45B7D1'], startangle=90)
        ax2.set_title('Distribution des Classes (%)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/01_class_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n📊 INTERPRÉTATION - Distribution des classes:")
        print("• Dataset relativement équilibré avec une légère prédominance de la classe 1")
        print("• Classe 1: 33.1% | Classe 2: 39.9% | Classe 0: 27.0%")
        print("• Pas de déséquilibre majeur qui nécessiterait un rééchantillonnage")
        
    def plot_feature_distributions(self):
        """Distribution des caractéristiques par classe"""
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        axes = axes.ravel()
        
        for i, feature in enumerate(self.feature_names):
            if i < len(axes):
                for class_name in self.target_names:
                    data = self.df[self.df['class_name'] == class_name][feature]
                    axes[i].hist(data, alpha=0.7, label=class_name, bins=15)
                
                axes[i].set_title(f'{feature}', fontsize=10)
                axes[i].set_xlabel('Valeur')
                axes[i].set_ylabel('Fréquence')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        # Supprime le subplot vide
        if len(self.feature_names) < len(axes):
            fig.delaxes(axes[-1])
        
        plt.suptitle('Distribution des Caractéristiques Chimiques par Classe', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/02_feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n📊 INTERPRÉTATION - Distributions des caractéristiques:")
        print("• Certaines caractéristiques montrent des différences claires entre classes")
        print("• Flavanoids et Proline semblent être de bons discriminants")
        print("• Quelques caractéristiques ont des distributions qui se chevauchent")
        
    def plot_correlation_matrix(self):
        """Matrice de corrélation des caractéristiques"""
        plt.figure(figsize=(15, 12))
        
        correlation_matrix = self.df[self.feature_names].corr()
        
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdYlBu_r', 
                   center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8},
                   fmt='.2f')
        
        plt.title('Matrice de Corrélation des Caractéristiques Chimiques', 
                 fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/03_correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Identification des corrélations fortes
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if abs(correlation_matrix.iloc[i, j]) > 0.7:
                    high_corr_pairs.append((
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        correlation_matrix.iloc[i, j]
                    ))
        
        print("\n📊 INTERPRÉTATION - Corrélations:")
        print("• Corrélations fortes (>0.7) identifiées:")
        for feat1, feat2, corr in high_corr_pairs:
            print(f"  - {feat1} ↔ {feat2}: {corr:.3f}")
        print("• Ces corrélations suggèrent des redondances potentielles")
        print("• Pourrait justifier une réduction de dimensionnalité")
        
    def plot_boxplots_by_class(self):
        """Box plots de TOUTES les caractéristiques par classe"""
        # Utilisation de toutes les caractéristiques
        all_features = self.feature_names
        
        # Calcul de la grille optimale (4 colonnes pour un bon affichage)
        n_features = len(all_features)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols  # Division avec arrondi vers le haut
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        
        # Si une seule ligne, axes n'est pas un array 2D
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        axes = axes.ravel()
        
        # Palette de couleurs pour les classes
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, feature in enumerate(all_features):
            if i < len(axes):
                # Box plot avec palette personnalisée
                box_plot = sns.boxplot(data=self.df, x='class_name', y=feature, 
                                      ax=axes[i], palette=colors)
                
                # Amélioration de l'apparence
                axes[i].set_title(f'{feature}'.replace('_', ' ').title(), 
                                 fontsize=11, fontweight='bold')
                axes[i].set_xlabel('Classe de Vin', fontsize=9)
                axes[i].set_ylabel('Valeur', fontsize=9)
                axes[i].grid(True, alpha=0.3)
                axes[i].tick_params(axis='both', which='major', labelsize=8)
                
                # Rotation des labels x si nécessaire
                axes[i].tick_params(axis='x', rotation=0)
        
        # Supprime les subplots vides
        for i in range(len(all_features), len(axes)):
            fig.delaxes(axes[i])
        
        plt.suptitle('Distribution de TOUTES les Caractéristiques Chimiques par Classe', 
                    fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Ajustement pour le titre
        plt.savefig(f'{self.save_dir}/04_boxplots_all_features.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Analyse détaillée de toutes les caractéristiques
        print("\n📊 INTERPRÉTATION DÉTAILLÉE - Box plots de toutes les caractéristiques:")
        print("="*80)
        
        # Calcul des moyennes par classe pour chaque caractéristique
        discrimination_scores = []
        
        for feature in all_features:
            means_by_class = []
            for class_name in self.target_names:
                class_data = self.df[self.df['class_name'] == class_name][feature]
                means_by_class.append(class_data.mean())
            
            # Calcul du coefficient de variation inter-classe
            cv_inter_class = np.std(means_by_class) / np.mean(means_by_class) if np.mean(means_by_class) != 0 else 0
            discrimination_scores.append((feature, cv_inter_class, means_by_class))
        
        # Tri par pouvoir discriminant
        discrimination_scores.sort(key=lambda x: x[1], reverse=True)
        
        print("\n TOP 10 DES CARACTÉRISTIQUES LES PLUS DISCRIMINANTES:")
        print("-" * 60)
        for i, (feature, score, means) in enumerate(discrimination_scores[:10], 1):
            class_order = np.argsort(means)[::-1]  # Ordre décroissant
            class_ranking = " > ".join([f"Classe {class_order[j]}" for j in range(3)])
            print(f"{i:2d}. {feature:<25} | Score: {score:.3f} | {class_ranking}")
        
        print(f"\n OBSERVATIONS CLÉS:")
        best_features = [item[0] for item in discrimination_scores[:5]]
        print(f"• Meilleures caractéristiques: {', '.join(best_features)}")
        print("• Ces caractéristiques montrent des différences nettes entre classes")
        print("• Parfaites pour construire un modèle de classification robuste")
        
        worst_features = [item[0] for item in discrimination_scores[-3:]]
        print(f"• Caractéristiques moins discriminantes: {', '.join(worst_features)}")
        print("• Ces variables ont des distributions similaires entre classes")
        
        print(f"\n RECOMMANDATIONS:")
        print(f"• Utiliser prioritairement les {len(best_features)} meilleures caractéristiques")
        print("• Envisager d'écarter les caractéristiques les moins discriminantes")
        print("• Appliquer une standardisation avant la modélisation (échelles différentes)")
        
        return discrimination_scores
        
    def plot_pca_analysis(self):
      
        # Standardisation des données
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.df[self.feature_names])
        
        # PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Variance expliquée
        cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
        ax1.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
               pca.explained_variance_ratio_)
        ax1.set_title('Variance Expliquée par Composante', fontweight='bold')
        ax1.set_xlabel('Composante Principale')
        ax1.set_ylabel('Variance Expliquée')
        ax1.grid(True, alpha=0.3)
        
        # Variance cumulée
        ax2.plot(range(1, len(cumsum_variance) + 1), cumsum_variance, 'bo-')
        ax2.axhline(y=0.8, color='r', linestyle='--', label='80% variance')
        ax2.axhline(y=0.95, color='orange', linestyle='--', label='95% variance')
        ax2.set_title('Variance Cumulée', fontweight='bold')
        ax2.set_xlabel('Nombre de Composantes')
        ax2.set_ylabel('Variance Cumulée')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Projection 2D (PC1 vs PC2)
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        for i, class_name in enumerate(self.target_names):
            mask = self.df['target'] == i
            ax3.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=colors[i], label=class_name, alpha=0.7)
        ax3.set_title('Projection PCA (PC1 vs PC2)', fontweight='bold')
        ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Projection 3D simulée (PC1 vs PC3)
        for i, class_name in enumerate(self.target_names):
            mask = self.df['target'] == i
            ax4.scatter(X_pca[mask, 0], X_pca[mask, 2], 
                       c=colors[i], label=class_name, alpha=0.7)
        ax4.set_title('Projection PCA (PC1 vs PC3)', fontweight='bold')
        ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax4.set_ylabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%} variance)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/05_pca_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n📊 INTERPRÉTATION - Analyse PCA:")
        print(f"• PC1 explique {pca.explained_variance_ratio_[0]:.1%} de la variance")
        print(f"• PC2 explique {pca.explained_variance_ratio_[1]:.1%} de la variance")
        print(f"• Les 3 premières composantes expliquent {cumsum_variance[2]:.1%} de la variance")
        print("• Séparation claire des classes dans l'espace PCA")
        print("• Réduction de dimensionnalité très efficace possible")
     
    def plot_feature_importance(self):
        """Importance des caractéristiques basée sur la variance inter-classe"""
        # Calcul de la variance inter-classe pour chaque caractéristique
        feature_importance = []
        
        for feature in self.feature_names:
            class_means = []
            for target in range(3):
                class_data = self.df[self.df['target'] == target][feature]
                class_means.append(class_data.mean())
            
            # Variance inter-classe
            inter_class_var = np.var(class_means)
            feature_importance.append((feature, inter_class_var))
        
        # Tri par importance
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        # Graphique
        features, importances = zip(*feature_importance)
        
        plt.figure(figsize=(12, 10))
        bars = plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Variance Inter-classe')
        plt.title('Importance des Caractéristiques (basée sur la variance inter-classe)', 
                 fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Coloration des barres
        colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/06_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n📊 INTERPRÉTATION - Importance des caractéristiques:")
        print("• Top 5 des caractéristiques les plus discriminantes:")
        for i, (feature, importance) in enumerate(feature_importance[:5]):
            print(f"  {i+1}. {feature}: {importance:.3f}")
        print("• Ces caractéristiques devraient être prioritaires pour la classification")
        
        
    def run_full_exploration(self):
        """Lance l'exploration complète"""
        print(" DÉMARRAGE DE L'EXPLORATION DU DATASET WINE")
        print("=" * 60)
        
        # Informations de base
        self.basic_info()
        
        # Génération des graphiques
        print("\n Génération des visualisations...")
        self.plot_class_distribution()
        self.plot_feature_distributions()
        self.plot_correlation_matrix()
        self.plot_boxplots_by_class()
        self.plot_pca_analysis()
        self.plot_feature_importance()
        
        
        
        print(f"\n Exploration terminée! Tous les graphiques sont sauvegardés dans '{self.save_dir}/'")
        print("\nFichiers générés:")
        for i, filename in enumerate([
            "01_class_distribution.png",
            "02_feature_distributions.png", 
            "03_correlation_matrix.png",
            "04_boxplots_key_features.png",
            "05_pca_analysis.png",
            "06_feature_importance.png",
           
        ], 1):
            print(f"  {i}. {filename}")

# UTILISATION
if __name__ == "__main__":
    # Créer l'explorateur et lancer l'analyse complète
    explorer = WineExplorer(save_dir="wine_analysis_plots")
    explorer.run_full_exploration()
