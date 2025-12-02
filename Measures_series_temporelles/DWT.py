import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pywt

class DWT_Haar:
    """Implémentation pédagogique de la DWT avec ondelette de Haar."""
    
    def __init__(self):
        self.wavelet_name = 'haar'
        
    def decompose(self, signal_data, level=None):
        """
        Décompose un signal avec la DWT.
        
        Parameters:
        -----------
        signal_data : array 1D, signal à décomposer
        level : int, nombre de niveaux de décomposition (None = max possible)
        
        Returns:
        --------
        coeffs : dict, coefficients DWT {'approx': [], 'details': []}
        """
        signal_data = np.array(signal_data, dtype=float)
        n = len(signal_data)
        
        # Vérifier que la longueur est une puissance de 2
        if not self._is_power_of_two(n):
            # Padding avec zéros
            new_n = 2**int(np.ceil(np.log2(n)))
            padded = np.zeros(new_n)
            padded[:n] = signal_data
            signal_data = padded
            n = new_n
        
        # Déterminer le niveau maximum
        max_level = int(np.log2(n))
        if level is None or level > max_level:
            level = max_level
        
        # Initialiser les coefficients
        coeffs = {'approx': [], 'details': []}
        current_signal = signal_data.copy()
        
        print(f"Décomposition DWT - Niveau {level}")
        print("-" * 50)
        
        for l in range(level):
            print(f"\nNiveau {l+1}:")
            print(f"Signal d'entrée: {current_signal}")
            
            # Longueur du signal actuel
            m = len(current_signal)
            
            # Calculer les approximations et détails
            approx = np.zeros(m // 2)
            details = np.zeros(m // 2)
            
            for i in range(0, m, 2):
                if i+1 < m:
                    idx = i // 2
                    # Approximation = moyenne
                    approx[idx] = (current_signal[i] + current_signal[i+1]) / np.sqrt(2)
                    # Détail = différence
                    details[idx] = (current_signal[i] - current_signal[i+1]) / np.sqrt(2)
            
            print(f"Approximation A{l+1}: {approx}")
            print(f"Détails D{l+1}: {details}")
            
            # Stocker les coefficients
            coeffs['details'].append(details)
            coeffs['approx'] = approx  # On ne garde que la dernière approximation
            
            # Pour le niveau suivant, on travaille sur l'approximation
            current_signal = approx
        
        print("-" * 50)
        return coeffs
    
    def reconstruct(self, coeffs):
        """
        Reconstruit le signal à partir des coefficients DWT.
        
        Parameters:
        -----------
        coeffs : dict, coefficients DWT
        
        Returns:
        --------
        reconstructed : array, signal reconstruit
        """
        # Commencer par l'approximation la plus grossière
        approx = coeffs['approx']
        details_list = coeffs['details']
        
        print(f"\nReconstruction DWT")
        print("-" * 50)
        
        # Reconstruire niveau par niveau
        for l in range(len(details_list)-1, -1, -1):
            details = details_list[l]
            
            print(f"\nNiveau {l+1}:")
            print(f"Approximation entrante: {approx}")
            print(f"Détails: {details}")
            
            m = len(approx)
            reconstructed = np.zeros(2 * m)
            
            for i in range(m):
                # Formule de reconstruction inverse pour Haar
                reconstructed[2*i] = (approx[i] + details[i]) / np.sqrt(2)
                reconstructed[2*i+1] = (approx[i] - details[i]) / np.sqrt(2)
            
            print(f"Signal reconstruit: {reconstructed}")
            approx = reconstructed
        
        print("-" * 50)
        return approx
    
    def _is_power_of_two(self, n):
        """Vérifie si n est une puissance de 2."""
        return (n & (n-1) == 0) and n != 0

# Exemple avec notre signal manuel
def example_manual_dwt():
    """Exemple de DWT avec notre signal [1,2,3,4,5,6,7,8]."""
    
    print("=" * 60)
    print("EXEMPLE MANUEL DWT - ONDELETTE DE HAAR")
    print("=" * 60)
    
    # Signal d'exemple
    signal_data = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    
    # Créer l'objet DWT
    dwt = DWT_Haar()
    
    # Décomposition
    print(f"\nSignal original: {signal_data}")
    coeffs = dwt.decompose(signal_data, level=3)
    
    # Reconstruction
    reconstructed = dwt.reconstruct(coeffs)
    
    # Vérification
    print(f"\nSignal original:    {signal_data}")
    print(f"Signal reconstruit: {reconstructed}")
    print(f"Erreur: {np.max(np.abs(signal_data - reconstructed[:len(signal_data)]))}")
    
    return coeffs, reconstructed

# Version 2 : Implémentation avec PyWavelets (bibliothèque standard)
def dwt_pywavelets_example():
    """Exemple avec la bibliothèque PyWavelets."""
    import pywt
    
    print("\n" + "=" * 60)
    print("DWT AVEC PYWAVELETS")
    print("=" * 60)
    
    # Signal d'exemple
    signal_data = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    
    # Décomposition avec ondelette de Haar
    coeffs = pywt.wavedec(signal_data, 'haar', level=3)
    
    print(f"Signal original: {signal_data}")
    print(f"\nCoefficients DWT (format pywt):")
    
    # Afficher les coefficients
    for i, c in enumerate(coeffs):
        if i == 0:
            print(f"  Approximation niveau 3 (A3): {c}")
        else:
            print(f"  Détails niveau {4-i} (D{4-i}): {c}")
    
    # Reconstruction
    reconstructed = pywt.waverec(coeffs, 'haar')
    
    print(f"\nSignal reconstruit: {reconstructed}")
    print(f"Erreur: {np.max(np.abs(signal_data - reconstructed))}")
    
    return coeffs, reconstructed

# Exemple 3 : DWT pour la compression d'images
def dwt_image_compression():
    """Exemple de compression d'image avec DWT."""
    import matplotlib.image as mpimg
    
    print("\n" + "=" * 60)
    print("COMPRESSION D'IMAGE AVEC DWT")
    print("=" * 60)
    
    # Créer une image synthétique simple
    np.random.seed(42)
    image = np.zeros((64, 64))
    
    # Ajouter des motifs
    image[10:30, 10:30] = 1.0  # Carré blanc
    image[40:60, 40:60] = 0.5  # Carré gris
    
    # Ajouter du bruit
    image += np.random.normal(0, 0.1, image.shape)
    
    # Appliquer la DWT 2D
    coeffs2 = pywt.dwt2(image, 'haar')
    cA, (cH, cV, cD) = coeffs2
    
    # Compression : Seuil des coefficients
    threshold = 0.2
    cA_thresh = cA * (np.abs(cA) > threshold)
    cH_thresh = cH * (np.abs(cH) > threshold)
    cV_thresh = cV * (np.abs(cV) > threshold)
    cD_thresh = cD * (np.abs(cD) > threshold)
    
    # Reconstruction
    compressed = pywt.idwt2((cA_thresh, (cH_thresh, cV_thresh, cD_thresh)), 'haar')
    
    # Calculer la compression
    n_original = image.size
    n_compressed = np.sum(cA_thresh != 0) + np.sum(cH_thresh != 0) + \
                   np.sum(cV_thresh != 0) + np.sum(cD_thresh != 0)
    
    compression_ratio = n_original / n_compressed
    
    # Visualisation
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Image originale
    axes[0, 0].imshow(image, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Image originale')
    axes[0, 0].axis('off')
    
    # Coefficients DWT
    axes[0, 1].imshow(cA, cmap='gray')
    axes[0, 1].set_title('Coefficients Approximation')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(cH, cmap='gray')
    axes[0, 2].set_title('Coefficients Horizontaux')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(cV, cmap='gray')
    axes[1, 0].set_title('Coefficients Verticaux')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(cD, cmap='gray')
    axes[1, 1].set_title('Coefficients Diagonaux')
    axes[1, 1].axis('off')
    
    # Image reconstruite
    axes[1, 2].imshow(compressed, cmap='gray', vmin=0, vmax=1)
    axes[1, 2].set_title(f'Image compressée\nRatio: {compression_ratio:.1f}:1')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Pixels originaux: {n_original}")
    print(f"Pixels non nuls après compression: {n_compressed}")
    print(f"Ratio de compression: {compression_ratio:.1f}:1")
    
    return image, compressed, compression_ratio

# Exemple 4 : DWT pour le débruitage de signal
def dwt_denoising():
    """Exemple de débruitage de signal avec DWT."""
    print("\n" + "=" * 60)
    print("DÉBRUITAGE DE SIGNAL AVEC DWT")
    print("=" * 60)
    
    # Créer un signal propre
    t = np.linspace(0, 1, 256)
    clean_signal = np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*20*t)
    
    # Ajouter du bruit
    np.random.seed(42)
    noise = np.random.normal(0, 0.3, len(t))
    noisy_signal = clean_signal + noise
    
    # Décomposition DWT
    coeffs = pywt.wavedec(noisy_signal, 'db4', level=4)
    
    # Seuillage des coefficients (débruitage)
    threshold = 0.5
    coeffs_thresh = []
    coeffs_thresh.append(coeffs[0])  # Garder l'approximation
    
    for i in range(1, len(coeffs)):
        # Seuillage dur
        coeff_thresh = pywt.threshold(coeffs[i], threshold, mode='hard')
        coeffs_thresh.append(coeff_thresh)
    
    # Reconstruction
    denoised_signal = pywt.waverec(coeffs_thresh, 'db4')
    
    # Calcul des erreurs
    noise_error = np.mean((noisy_signal - clean_signal)**2)
    denoised_error = np.mean((denoised_signal - clean_signal)**2)
    
    # Visualisation
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    
    # Signal propre
    axes[0].plot(t, clean_signal, 'b-', linewidth=2, label='Signal propre')
    axes[0].set_title('Signal propre (sans bruit)')
    axes[0].set_xlabel('Temps')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Signal bruité
    axes[1].plot(t, noisy_signal, 'r-', alpha=0.7, label='Signal bruité')
    axes[1].plot(t, clean_signal, 'b--', linewidth=1, label='Signal propre')
    axes[1].set_title(f'Signal bruité (Erreur MSE: {noise_error:.4f})')
    axes[1].set_xlabel('Temps')
    axes[1].set_ylabel('Amplitude')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Signal débruité
    axes[2].plot(t, denoised_signal, 'g-', linewidth=2, label='Signal débruité')
    axes[2].plot(t, clean_signal, 'b--', linewidth=1, label='Signal propre')
    axes[2].set_title(f'Signal après débruitage DWT (Erreur MSE: {denoised_error:.4f})')
    axes[2].set_xlabel('Temps')
    axes[2].set_ylabel('Amplitude')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Erreur MSE (bruité): {noise_error:.4f}")
    print(f"Erreur MSE (débruité): {denoised_error:.4f}")
    print(f"Amélioration: {100*(1 - denoised_error/noise_error):.1f}%")
    
    return clean_signal, noisy_signal, denoised_signal

# Exemple 5 : DWT pour l'analyse de fréquences
def dwt_frequency_analysis():
    """Exemple d'analyse fréquentielle avec DWT."""
    print("\n" + "=" * 60)
    print("ANALYSE FRÉQUENTIELLE AVEC DWT")
    print("=" * 60)
    
    # Créer un signal composite
    fs = 1000  # Fréquence d'échantillonnage (Hz)
    t = np.arange(0, 1, 1/fs)
    
    # Composantes fréquentielles
    f1 = 5   # Hz (basse fréquence)
    f2 = 50  # Hz (moyenne fréquence)
    f3 = 150 # Hz (haute fréquence)
    
    signal_data = (np.sin(2*np.pi*f1*t) + 
                   0.5*np.sin(2*np.pi*f2*t) + 
                   0.2*np.sin(2*np.pi*f3*t))
    
    # Décomposition DWT (4 niveaux)
    coeffs = pywt.wavedec(signal_data, 'db4', level=4)
    
    # Fréquences approximatives pour chaque niveau
    # Niveau 1 : ~250-500 Hz, Niveau 2 : ~125-250 Hz, etc.
    freq_bands = [
        (250, 500),   # D1
        (125, 250),   # D2
        (62.5, 125),  # D3
        (31.25, 62.5),# D4
        (0, 31.25)    # A4 (approximation)
    ]
    
    # Visualisation
    fig, axes = plt.subplots(len(coeffs)+1, 1, figsize=(12, 10))
    
    # Signal original
    axes[0].plot(t, signal_data, 'b-')
    axes[0].set_title('Signal original (composantes à 5, 50 et 150 Hz)')
    axes[0].set_xlabel('Temps (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    
    # Coefficients DWT
    for i, c in enumerate(coeffs):
        if i == 0:
            title = f'Approximation A{len(coeffs)-i} ({freq_bands[-1][0]}-{freq_bands[-1][1]} Hz)'
        else:
            level = len(coeffs) - i
            title = f'Détails D{level} ({freq_bands[level-1][0]}-{freq_bands[level-1][1]} Hz)'
        
        # Créer un vecteur temps pour les coefficients (sous-échantillonnés)
        t_coeff = np.linspace(0, 1, len(c))
        
        axes[i+1].plot(t_coeff, c, 'r-')
        axes[i+1].set_title(title)
        axes[i+1].set_xlabel('Temps normalisé')
        axes[i+1].set_ylabel('Amplitude')
        axes[i+1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Analyse énergétique
    print("\nAnalyse énergétique par bande de fréquence:")
    print("-" * 50)
    
    total_energy = np.sum(signal_data**2)
    
    for i, c in enumerate(coeffs):
        if i == 0:
            band_name = f"A{len(coeffs)}"
            freq_range = f"{freq_bands[-1][0]}-{freq_bands[-1][1]} Hz"
        else:
            level = len(coeffs) - i
            band_name = f"D{level}"
            freq_range = f"{freq_bands[level-1][0]}-{freq_bands[level-1][1]} Hz"
        
        band_energy = np.sum(c**2)
        percentage = (band_energy / total_energy) * 100
        
        print(f"{band_name:5} ({freq_range:15}) : {percentage:6.2f}%")
    
    return signal_data, coeffs, freq_bands

# Fonction principale
def main():
    """Exécute tous les exemples DWT."""
    print("=" * 60)
    print("TRANSFORMÉE EN ONDELETTES DISCRÈTE (DWT)")
    print("=" * 60)
    
    # Installation nécessaire
    print("\nInstallation requise:")
    print("pip install numpy matplotlib scipy PyWavelets")
    
    try:
        import pywt
        print("✓ PyWavelets est installé")
    except ImportError:
        print("\n⚠ PyWavelets n'est pas installé")
        print("Installez-le avec: pip install PyWavelets")
        return
    
    # Exemple 1 : Manuel
    print("\n1. Exemple manuel avec ondelette de Haar")
    coeffs_manual, reconstructed_manual = example_manual_dwt()
    
    # Exemple 2 : PyWavelets
    print("\n2. Exemple avec PyWavelets")
    coeffs_pywt, reconstructed_pywt = dwt_pywavelets_example()
    
    # Exemple 3 : Compression d'image
    print("\n3. Compression d'image avec DWT")
    image, compressed, ratio = dwt_image_compression()
    
    # Exemple 4 : Débruitage
    print("\n4. Débruitage de signal avec DWT")
    clean, noisy, denoised = dwt_denoising()
    
    # Exemple 5 : Analyse fréquentielle
    print("\n5. Analyse fréquentielle avec DWT")
    signal, coeffs_freq, freq_bands = dwt_frequency_analysis()
    
    print("\n" + "=" * 60)
    print("DÉMONSTRATION TERMINÉE")
    print("=" * 60)

if __name__ == "__main__":
    main()
