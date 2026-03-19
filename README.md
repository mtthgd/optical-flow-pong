# 🎮 Optical Flow Pong

**Optical Flow Pong** est un projet de computer vision, reprenant le jeu Pong est ou la raquette devient contrôlable par le doigt capté par une webcam.

Le système repose sur deux briques principales :
- MediaPipe pour la détection et le suivi du doigt
- Optical Flow (Farnebäck) pour estimer le mouvement



# 📌 Objectif

Créer une interaction temps réel entre un utilisateur et un jeu.

Le pipeline est le suivant :
1. Capture vidéo webcam
2. Détection de la main
3. Extraction de l’index
4. Calcul du mouvement avec optical flow
5. Conversion en signal de contrôle
6. Interaction avec le jeu Pong



# 🎥 Démonstration


Section qui sera complétée ultérieurement 



# ⚙️ Fonctionnalités

- Tracking du doigt avec MediaPipe Hands
- Optical flow dense avec Farnebäck
- Estimation de vitesse en temps réel (px/s)
- Lissage du signal (EMA)
- Contrôle de la raquette
- Effet smash basé sur la vitesse
- IA simple pour l’adversaire
- Affichage FPS, vitesse
- Visualisation du champ de flow



# 🔄 Pipeline

Webcam  
↓  
Image acquisition  
↓  
MediaPipe (détection main)  
↓  
Extraction landmark (index)  
↓  
ROI autour du doigt  
↓  
Optical Flow (Farnebäck)  
↓  
Estimation vitesse  
↓  
Lissage  
↓  
Contrôle du jeu  



# 🧠 Méthodes utilisées

## Optical Flow

Hypothèse de constance de luminosité :

I(x, y, t) = I(x + dx, y + dy, t + dt)

Formulation :

Ix * u + Iy * v + It = 0

avec :
- (u, v) = vecteur de mouvement
- Ix, Iy = gradients spatiaux
- It = gradient temporel


### Farnebäck (choisi)

- Approximation polynomiale locale
- Optical flow dense
- Meilleure robustesse aux mouvements complexes

Utilisé dans ce projet pour produire un signal de contrôle fluide.



# ✋ MediaPipe

MediaPipe permet de :
- détecter la main
- extraire les landmarks et donc suivre l'index 

Cela permet de limiter le calcul de l’optical flow à une zone d’intérêt (ROI), ce qui :
- réduit le bruit
- améliore les performances
- évite les artefacts liés au fond



# 🏗️ Architecture

optical-flow-pong/  
├── README.md  
├── hand_landmarker.task  
├── pyproject.toml  
└── src/  
  ├── main.py  
  ├── game/  
  │  └── pong.py  
  └── vision/  
    └── finger_flow_tracker.py  



# 🛠️ Installation

## Prérequis
- Python 3.11+
- Webcam
- uv (gestionnaire de packages)

Installation de uv (si nécessaire) :
```curl -Ls https://astral.sh/uv/install.sh | sh```


## Installation du projet

```git clone https://github.com/mtthgd/optical-flow-pong.git```
```cd optical-flow-pong ``` 



## Installation des dépendances

```uv sync```


## Activation de l’environnement

```source .venv/bin/activate```


# ▶️ Lancement

Pour lancer le jeu :

```uv run python -m src.game.pong```  (appuyer sur la touche ESC pour quitter)

Pour lancer la visualisation du tracking du doigt + optical flow : 

```uv run python -m src.vision.finger_flow_tracker.py``` (appuyer sur la touche Q pour quitter)



# 🎮 Utilisation

- Se placer devant la webcam  
- Montrer sa main  
- Utiliser l’index pour contrôler la raquette  
- Bouger rapidement pour déclencher un smash  

---

# ⚙️ Paramètres importants

Dans le tracker :
- roi_radius : taille de la zone d’analyse
- ema_alpha : niveau de lissage
- flow_patch : zone d’estimation locale

Dans le jeu :
- SMASH_SPEED_THRESHOLD
- SMASH_MULT
- MAX_BALL_SPEED
- SPIN_FACTOR

---

# ⚠️ Limitations

- Sensible aux conditions de lumière
- Dépendance à la qualité du tracking MediaPipe
- Optical flow bruité dans certaines scènes lumineuses


--- 

Projet réalisé dans le cadre de l'UE COMPUTER VISION @ IMT ATLANTIQUE 


