import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class SimpleNN(nn.Module):
    """Réseau de neurones simple pour continual learning."""
    
    def __init__(self, input_size=3, hidden_size=4, output_size=2):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class EWCContinualLearner:
    """Apprentissage continu avec EWC."""
    
    def __init__(self, model, lr=0.01, ewc_lambda=1000):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.ewc_lambda = ewc_lambda
        
        # Pour stocker les poids importants des tâches précédentes
        self.importance_matrices = []  # Liste des F pour chaque tâche
        self.optimal_params = []       # Liste des θ* pour chaque tâche
        self.tasks = []                # Informations sur les tâches
        
    def compute_fisher_information(self, task_data, num_samples=100):
        """Calcule la matrice de Fisher (importance des poids)."""
        self.model.eval()
        
        # Initialiser les matrices F à zéro
        fisher_matrices = {}
        for name, param in self.model.named_parameters():
            fisher_matrices[name] = torch.zeros_like(param.data)
        
        # Échantillonner pour estimer Fisher
        for _ in range(num_samples):
            self.model.zero_grad()
            
            # Données aléatoires de la tâche
            x, y = task_data.sample_batch(batch_size=10)
            
            # Calculer la loss
            output = self.model(x)
            loss = nn.functional.cross_entropy(output, y)
            loss.backward()
            
            # Accumuler (gradient)²
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher_matrices[name] += param.grad.data ** 2 / num_samples
        
        return fisher_matrices
    
    def train_task(self, task_id, task_data, epochs=50, use_ewc=True):
        """Entraîne sur une nouvelle tâche."""
        print(f"\n{'='*50}")
        print(f"ENTRAÎNEMENT TÂCHE {task_id}")
        print('='*50)
        
        losses = []
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for x_batch, y_batch in task_data.get_batches(batch_size=10):
                self.model.zero_grad()
                
                # Forward pass
                outputs = self.model(x_batch)
                task_loss = nn.functional.cross_entropy(outputs, y_batch)
                
                # Ajouter la pénalité EWC pour les tâches précédentes
                ewc_loss = 0
                if use_ewc and len(self.importance_matrices) > 0:
                    for task_idx in range(len(self.importance_matrices)):
                        for name, param in self.model.named_parameters():
                            optimal_param = self.optimal_params[task_idx][name]
                            importance = self.importance_matrices[task_idx][name]
                            
                            # Pénalité quadratique
                            ewc_loss += (importance * (param - optimal_param) ** 2).sum()
                    
                    ewc_loss *= self.ewc_lambda / 2
                
                # Loss totale
                loss = task_loss + ewc_loss
                total_loss += loss.item()
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                num_batches += 1
            
            avg_loss = total_loss / max(num_batches, 1)
            losses.append(avg_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
        
        # Après entraînement, sauvegarder les poids et calculer Fisher
        optimal_params = {}
        for name, param in self.model.named_parameters():
            optimal_params[name] = param.data.clone()
        
        self.optimal_params.append(optimal_params)
        
        # Calculer l'importance (Fisher) pour cette tâche
        fisher_matrix = self.compute_fisher_information(task_data)
        self.importance_matrices.append(fisher_matrix)
        
        self.tasks.append({
            'id': task_id,
            'data_info': task_data.get_info()
        })
        
        return losses
    
    def evaluate(self, test_data_dict):
        """Évalue la performance sur toutes les tâches apprises."""
        print(f"\n{'='*50}")
        print("ÉVALUATION SUR TOUTES LES TÂCHES")
        print('='*50)
        
        results = {}
        self.model.eval()
        
        with torch.no_grad():
            for task_name, test_data in test_data_dict.items():
                correct = 0
                total = 0
                
                for x, y in test_data.get_all_data():
                    outputs = self.model(x)
                    _, predicted = torch.max(outputs.data, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
                
                accuracy = correct / total * 100
                results[task_name] = accuracy
                print(f"Tâche {task_name}: {accuracy:.2f}%")
        
        return results

class TaskData:
    """Classe pour gérer les données d'une tâche."""
    
    def __init__(self, data, labels, task_name):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.task_name = task_name
        self.num_samples = len(data)
        
    def sample_batch(self, batch_size=10):
        """Échantillonne un batch aléatoire."""
        indices = torch.randint(0, self.num_samples, (batch_size,))
        return self.data[indices], self.labels[indices]
    
    def get_batches(self, batch_size=10):
        """Générateur de batches."""
        indices = torch.randperm(self.num_samples)
        for i in range(0, self.num_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            yield self.data[batch_indices], self.labels[batch_indices]
    
    def get_all_data(self):
        """Retourne toutes les données."""
        yield self.data, self.labels
    
    def get_info(self):
        return f"{self.task_name} ({self.num_samples} samples)"

# Exemple 1 : MNIST simplifié
def example_simplified_mnist():
    """Exemple avec digits MNIST simplifiés."""
    
    print("=" * 60)
    print("CONTINUAL LEARNING - EXEMPLE MNIST SIMPLIFIÉ")
    print("=" * 60)
    
    # Créer des données simplifiées pour 3 tâches
    np.random.seed(42)
    
    # Tâche 1: Digits 0 et 1
    task1_data = []
    task1_labels = []
    
    for _ in range(100):
        # Digit 0: [0, 0, 1]
        task1_data.append([0, 0, 1])
        task1_labels.append(0)
        
        # Digit 1: [0, 1, 0]
        task1_data.append([0, 1, 0])
        task1_labels.append(1)
    
    task1 = TaskData(task1_data, task1_labels, "Digits 0-1")
    
    # Tâche 2: Digits 2 et 3
    task2_data = []
    task2_labels = []
    
    for _ in range(100):
        # Digit 2: [1, 0, 0]
        task2_data.append([1, 0, 0])
        task2_labels.append(0)  # Classe 0 pour cette tâche
        
        # Digit 3: [1, 1, 0]
        task2_data.append([1, 1, 0])
        task2_labels.append(1)  # Classe 1 pour cette tâche
    
    task2 = TaskData(task2_data, task2_labels, "Digits 2-3")
    
    # Tâche 3: Digits 4 et 5
    task3_data = []
    task3_labels = []
    
    for _ in range(100):
        # Digit 4: [0, 0, 0]
        task3_data.append([0, 0, 0])
        task3_labels.append(0)
        
        # Digit 5: [1, 0, 1]
        task3_data.append([1, 0, 1])
        task3_labels.append(1)
    
    task3 = TaskData(task3_data, task3_labels, "Digits 4-5")
    
    # Données de test pour chaque tâche
    test_data = {
        "Tâche1": TaskData(task1_data[:20], task1_labels[:20], "Test 0-1"),
        "Tâche2": TaskData(task2_data[:20], task2_labels[:20], "Test 2-3"),
        "Tâche3": TaskData(task3_data[:20], task3_labels[:20], "Test 4-5")
    }
    
    # Test 1: Sans continual learning (oubli catastrophique)
    print("\n1. SANS CONTINUAL LEARNING (oubli catastrophique)")
    print("-" * 50)
    
    model_no_cl = SimpleNN(input_size=3, hidden_size=4, output_size=2)
    learner_no_cl = EWCContinualLearner(model_no_cl, ewc_lambda=0)  # λ=0 désactive EWC
    
    # Entraînement séquentiel
    learner_no_cl.train_task(1, task1, epochs=30, use_ewc=False)
    results_after_t1 = learner_no_cl.evaluate({"Tâche1": test_data["Tâche1"]})
    
    learner_no_cl.train_task(2, task2, epochs=30, use_ewc=False)
    results_after_t2 = learner_no_cl.evaluate({
        "Tâche1": test_data["Tâche1"],
        "Tâche2": test_data["Tâche2"]
    })
    
    learner_no_cl.train_task(3, task3, epochs=30, use_ewc=False)
    results_after_t3 = learner_no_cl.evaluate({
        "Tâche1": test_data["Tâche1"],
        "Tâche2": test_data["Tâche2"],
        "Tâche3": test_data["Tâche3"]
    })
    
    # Test 2: Avec EWC
    print("\n\n2. AVEC EWC (Continual Learning)")
    print("-" * 50)
    
    model_ewc = SimpleNN(input_size=3, hidden_size=4, output_size=2)
    learner_ewc = EWCContinualLearner(model_ewc, ewc_lambda=1000)
    
    # Entraînement séquentiel avec EWC
    learner_ewc.train_task(1, task1, epochs=30, use_ewc=True)
    results_ewc_t1 = learner_ewc.evaluate({"Tâche1": test_data["Tâche1"]})
    
    learner_ewc.train_task(2, task2, epochs=30, use_ewc=True)
    results_ewc_t2 = learner_ewc.evaluate({
        "Tâche1": test_data["Tâche1"],
        "Tâche2": test_data["Tâche2"]
    })
    
    learner_ewc.train_task(3, task3, epochs=30, use_ewc=True)
    results_ewc_t3 = learner_ewc.evaluate({
        "Tâche1": test_data["Tâche1"],
        "Tâche2": test_data["Tâche2"],
        "Tâche3": test_data["Tâche3"]
    })
    
    # Visualisation comparative
    print(f"\n{'='*60}")
    print("COMPARAISON DES PERFORMANCES")
    print('='*60)
    
    tasks = ["Tâche1", "Tâche2", "Tâche3"]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Graphique 1: Performance après chaque apprentissage
    x = np.arange(3)
    width = 0.35
    
    # Sans CL
    no_cl_scores = [
        results_after_t1.get("Tâche1", 0),
        results_after_t2.get("Tâche2", 0),
        results_after_t3.get("Tâche3", 0)
    ]
    
    # Avec EWC
    ewc_scores = [
        results_ewc_t1.get("Tâche1", 0),
        results_ewc_t2.get("Tâche2", 0),
        results_ewc_t3.get("Tâche3", 0)
    ]
    
    axes[0].bar(x - width/2, no_cl_scores, width, label='Sans CL', color='red', alpha=0.7)
    axes[0].bar(x + width/2, ewc_scores, width, label='Avec EWC', color='green', alpha=0.7)
    
    axes[0].set_xlabel('Tâche')
    axes[0].set_ylabel('Précision (%)')
    axes[0].set_title('Performance sur la tâche récemment apprise')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(['Tâche 1', 'Tâche 2', 'Tâche 3'])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Graphique 2: Oubli des tâches précédentes
    axes[1].plot([1, 2, 3], 
                 [results_after_t1.get("Tâche1", 0),
                  results_after_t2.get("Tâche1", 0),
                  results_after_t3.get("Tâche1", 0)],
                 'r-o', label='Sans CL', linewidth=2)
    
    axes[1].plot([1, 2, 3],
                 [results_ewc_t1.get("Tâche1", 0),
                  results_ewc_t2.get("Tâche1", 0),
                  results_ewc_t3.get("Tâche1", 0)],
                 'g-s', label='Avec EWC', linewidth=2)
    
    axes[1].set_xlabel('Nombre de tâches apprises')
    axes[1].set_ylabel('Précision Tâche 1 (%)')
    axes[1].set_title('Oubli de la Tâche 1 au fil du temps')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return learner_ewc, learner_no_cl

# Version 2 : Replay Memory
class ReplayMemoryLearner:
    """Continual Learning avec mémoire de replay."""
    
    def __init__(self, model, memory_size=20, lr=0.01):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.memory_size = memory_size
        self.memory = {}  # task_id -> (data, labels)
        self.tasks_learned = []
        
    def store_in_memory(self, task_id, data, labels, num_samples=10):
        """Stocke des exemples dans la mémoire."""
        indices = np.random.choice(len(data), min(num_samples, len(data)), replace=False)
        
        memory_data = data[indices]
        memory_labels = labels[indices]
        
        self.memory[task_id] = (memory_data, memory_labels)
    
    def train_task(self, task_id, task_data, epochs=50):
        """Entraîne sur une nouvelle tâche avec replay."""
        print(f"\nEntraînement Tâche {task_id} avec replay memory...")
        
        # Stocker des exemples de la nouvelle tâche
        self.store_in_memory(task_id, task_data.data.numpy(), 
                            task_data.labels.numpy())
        self.tasks_learned.append(task_id)
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            # Mélanger replay et nouvelles données
            replay_ratio = 0.3  # 30% de replay, 70% de nouvelles données
            
            for x_new, y_new in task_data.get_batches(batch_size=16):
                self.model.zero_grad()
                
                # Données de la nouvelle tâche
                outputs_new = self.model(x_new)
                loss_new = nn.functional.cross_entropy(outputs_new, y_new)
                
                # Ajouter des données de replay
                loss_replay = 0
                if len(self.memory) > 0:
                    replay_batch_size = int(16 * replay_ratio)
                    
                    # Sélectionner aléatoirement des tâches précédentes
                    replay_tasks = np.random.choice(
                        list(self.memory.keys()), 
                        min(replay_batch_size, len(self.memory)), 
                        replace=True
                    )
                    
                    replay_data = []
                    replay_labels = []
                    
                    for t in replay_tasks:
                        mem_data, mem_labels = self.memory[t]
                        idx = np.random.randint(0, len(mem_data))
                        replay_data.append(mem_data[idx])
                        replay_labels.append(mem_labels[idx])
                    
                    if replay_data:
                        x_replay = torch.FloatTensor(replay_data)
                        y_replay = torch.LongTensor(replay_labels)
                        
                        outputs_replay = self.model(x_replay)
                        loss_replay = nn.functional.cross_entropy(outputs_replay, y_replay)
                
                # Loss totale
                loss = loss_new + loss_replay
                total_loss += loss.item()
                
                loss.backward()
                self.optimizer.step()
                num_batches += 1
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss/max(num_batches,1):.4f}")

# Version 3 : Architecture Progressive (Progressive Neural Networks)
class ProgressiveNetwork(nn.Module):
    """Réseau neuronal progressif pour continual learning."""
    
    def __init__(self, input_size, hidden_size, output_size):
        super(ProgressiveNetwork, self).__init__()
        
        self.columns = nn.ModuleList()  # Une colonne par tâche
        self.adapters = nn.ModuleList()  # Connexions entre colonnes
        
        # Colonne initiale
        col1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        self.columns.append(col1)
        
    def add_column(self, input_size, hidden_size, output_size):
        """Ajoute une nouvelle colonne pour une nouvelle tâche."""
        new_col = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
        # Adapter des colonnes précédentes vers la nouvelle
        adapter = nn.ModuleList()
        for prev_col in self.columns:
            adapter_layer = nn.Linear(hidden_size, hidden_size)
            adapter.append(adapter_layer)
        
        self.columns.append(new_col)
        self.adapters.append(adapter)
    
    def forward(self, x, task_id):
        """Forward pass pour une tâche spécifique."""
        # Activation de la colonne de la tâche
        current_col = self.columns[task_id]
        output = current_col(x)
        
        # Si ce n'est pas la première colonne, ajouter les adapters
        if task_id > 0:
            for i in range(task_id):
                prev_output = self.columns[i](x)
                if isinstance(prev_output, tuple):
                    prev_output = prev_output[0]
                
                adapted = self.adapters[task_id-1][i](prev_output)
                output = output + adapted
        
        return output

# Exemple 2 : Apprentissage de fonctions
def function_learning_example():
    """Exemple d'apprentissage continu de fonctions mathématiques."""
    
    print("\n" + "=" * 60)
    print("CONTINUAL LEARNING DE FONCTIONS MATHÉMATIQUES")
    print("=" * 60)
    
    # Générer des données pour 3 fonctions
    np.random.seed(42)
    x = np.linspace(-5, 5, 100).reshape(-1, 1)
    
    # Tâche 1: sin(x)
    y1 = np.sin(x) + np.random.normal(0, 0.1, x.shape)
    
    # Tâche 2: x²
    y2 = x**2 + np.random.normal(0, 0.5, x.shape)
    
    # Tâche 3: sigmoid(x)
    y3 = 1 / (1 + np.exp(-x)) + np.random.normal(0, 0.05, x.shape)
    
    # Convertir en tensors
    x_tensor = torch.FloatTensor(x)
    y1_tensor = torch.FloatTensor(y1)
    y2_tensor = torch.FloatTensor(y2)
    y3_tensor = torch.FloatTensor(y3)
    
    # Créer les datasets
    class FunctionDataset:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            
        def get_batches(self, batch_size=32):
            indices = torch.randperm(len(self.x))
            for i in range(0, len(self.x), batch_size):
                batch_idx = indices[i:i+batch_size]
                yield self.x[batch_idx], self.y[batch_idx]
        
        def get_all(self):
            return self.x, self.y
    
    task1_data = FunctionDataset(x_tensor, y1_tensor)
    task2_data = FunctionDataset(x_tensor, y2_tensor)
    task3_data = FunctionDataset(x_tensor, y3_tensor)
    
    # Test 1: Sans continual learning
    print("\n1. APPRENTISSAGE SÉQUENTIEL SANS CL")
    
    model_no_cl = nn.Sequential(
        nn.Linear(1, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )
    
    optimizer = optim.Adam(model_no_cl.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # Entraînement séquentiel
    print("Apprentissage sin(x)...")
    for epoch in range(100):
        for x_batch, y_batch in task1_data.get_batches(32):
            optimizer.zero_grad()
            outputs = model_no_cl(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
    
    # Évaluer sin(x)
    with torch.no_grad():
        pred1 = model_no_cl(x_tensor)
        mse1 = criterion(pred1, y1_tensor).item()
        print(f"MSE sin(x) après apprentissage: {mse1:.4f}")
    
    print("\nApprentissage x²...")
    for epoch in range(100):
        for x_batch, y_batch in task2_data.get_batches(32):
            optimizer.zero_grad()
            outputs = model_no_cl(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
    
    # Réévaluer sin(x) (devrait être oublié)
    with torch.no_grad():
        pred1_after = model_no_cl(x_tensor)
        mse1_after = criterion(pred1_after, y1_tensor).item()
        print(f"MSE sin(x) après x²: {mse1_after:.4f} (oubli: {mse1_after/mse1:.1f}x pire!)")
    
    # Test 2: Avec replay memory
    print("\n\n2. AVEC REPLAY MEMORY")
    
    model_replay = nn.Sequential(
        nn.Linear(1, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )
    
    optimizer_replay = optim.Adam(model_replay.parameters(), lr=0.01)
    replay_memory = []
    
    # Fonction pour ajouter à la mémoire
    def add_to_memory(x, y, memory_size=50):
        replay_memory.append((x.clone(), y.clone()))
        if len(replay_memory) > memory_size:
            replay_memory.pop(0)
    
    print("Apprentissage sin(x) avec mémoire...")
    for epoch in range(100):
        for x_batch, y_batch in task1_data.get_batches(32):
            optimizer_replay.zero_grad()
            
            # Loss sur nouvelles données
            outputs = model_replay(x_batch)
            loss_new = criterion(outputs, y_batch)
            
            # Loss sur replay (si disponible)
            loss_replay = 0
            if replay_memory:
                x_replay, y_replay = replay_memory[np.random.randint(0, len(replay_memory))]
                outputs_replay = model_replay(x_replay.unsqueeze(0))
                loss_replay = criterion(outputs_replay, y_replay.unsqueeze(0))
            
            loss = loss_new + 0.5 * loss_replay
            loss.backward()
            optimizer_replay.step()
        
        # Ajouter à la mémoire
        add_to_memory(x_tensor[epoch % len(x_tensor)], y1_tensor[epoch % len(y1_tensor)])
    
    print("Apprentissage x² avec replay...")
    for epoch in range(100):
        for x_batch, y_batch in task2_data.get_batches(32):
            optimizer_replay.zero_grad()
            
            outputs = model_replay(x_batch)
            loss_new = criterion(outputs, y_batch)
            
            # Replay des deux tâches
            loss_replay = 0
            if replay_memory:
                for _ in range(5):  # 5 exemples de replay
                    x_replay, y_replay = replay_memory[np.random.randint(0, len(replay_memory))]
                    outputs_replay = model_replay(x_replay.unsqueeze(0))
                    loss_replay += criterion(outputs_replay, y_replay.unsqueeze(0))
                loss_replay /= 5
            
            loss = loss_new + 0.3 * loss_replay
            loss.backward()
            optimizer_replay.step()
        
        # Ajouter à la mémoire
        add_to_memory(x_tensor[epoch % len(x_tensor)], y2_tensor[epoch % len(y2_tensor)])
    
    # Évaluation finale
    with torch.no_grad():
        pred_sin_replay = model_replay(x_tensor)
        mse_sin_replay = criterion(pred_sin_replay, y1_tensor).item()
        
        pred_sq_replay = model_replay(x_tensor)
        mse_sq_replay = criterion(pred_sq_replay, y2_tensor).item()
        
        print(f"\nAvec replay memory:")
        print(f"  MSE sin(x): {mse_sin_replay:.4f}")
        print(f"  MSE x²: {mse_sq_replay:.4f}")
    
    # Visualisation
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    with torch.no_grad():
        # Sans CL
        axes[0].plot(x, y1, 'b.', alpha=0.5, label='Vrai sin(x)')
        axes[0].plot(x, model_no_cl(x_tensor).numpy(), 'r-', label='Prédiction')
        axes[0].set_title('Sans CL: sin(x) après x²')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Avec replay
        axes[1].plot(x, y1, 'b.', alpha=0.5, label='Vrai sin(x)')
        axes[1].plot(x, model_replay(x_tensor).numpy(), 'g-', label='Prédiction')
        axes[1].set_title('Avec replay: sin(x) après x²')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Comparaison x²
        axes[2].plot(x, y2, 'b.', alpha=0.5, label='Vrai x²')
        axes[2].plot(x, model_replay(x_tensor).numpy(), 'g-', label='Prédiction replay')
        axes[2].plot(x, model_no_cl(x_tensor).numpy(), 'r-', label='Prédiction sans CL')
        axes[2].set_title('Comparaison x²')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return model_no_cl, model_replay

# Fonction principale
def main():
    """Exécute tous les exemples de continual learning."""
    print("=" * 60)
    print("CONTINUAL LEARNING - APPRENTISSAGE CONTINU")
    print("=" * 60)
    
    # Exemple 1 : Classification simple
    print("\n1. EXEMPLE DE CLASSIFICATION (MNIST simplifié)")
    learner_ewc, learner_no_cl = example_simplified_mnist()
    
    # Exemple 2 : Régression de fonctions
    print("\n2. EXEMPLE DE RÉGRESSION (Fonctions mathématiques)")
    model_no_cl, model_replay = function_learning_example()
    
    print("\n" + "=" * 60)
    print("RÉSUMÉ DU CONTINUAL LEARNING")
    print("=" * 60)
    print("""
    Le Continual Learning vise à résoudre le problème de l'oubli catastrophique.
    
    Méthodes principales:
    1. Regularization-based (EWC, SI, MAS)
       • Pénalise les changements des poids importants
       • Exemple: EWC utilise la matrice de Fisher
    
    2. Replay-based (Replay Memory, GAN)
       • Rejoue des exemples des tâches précédentes
       • Nécessite une mémoire (limitée)
    
    3. Architecture-based (Progressive Nets, PackNet)
       • Ajoute de nouvelles capacités au réseau
       • Évite l'interférence mais fait grossir le modèle
    
    4. Optimization-based (GEM, A-GEM)
       • Contraint les mises à jour pour préserver les performances passées
    
    Métriques importantes:
    • Average Accuracy: Performance moyenne sur toutes les tâches
    • Forgetting Measure: Combien on oublie les anciennes tâches
    • Forward Transfer: Capacité à aider les futures tâches
    
    Applications:
    • Robots qui apprennent continuellement
    • Assistants personnels intelligents
    • Diagnostic médical évolutif
    • Véhicules autonomes
    
    Challenges:
    • Balance stabilité-plasticité
    • Mémoire limitée
    • Évaluation réaliste
    • Scalabilité
    """)

if __name__ == "__main__":
    main()
