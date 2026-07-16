# Service de transcription vocale monovoix

API REST de transcription audio en français basée sur OpenAI Whisper. Le service
convertit les fichiers reçus en WAV, exécute l'inférence sur le GPU lorsqu'il est
disponible, puis renvoie le texte complet et les segments horodatés.

## Application

### Fonctionnement

Au démarrage, l'application charge une seule fois le modèle Whisper configuré. Le
modèle `base` est utilisé par défaut et PyTorch sélectionne automatiquement CUDA si
un GPU compatible est accessible ; sinon, l'inférence s'exécute sur le CPU.

Pour chaque transcription :

1. FastAPI reçoit le fichier dans le champ multipart `audioFile` ;
2. FFmpeg convertit le fichier en WAV temporaire ;
3. Whisper transcrit le WAV en forçant la langue française ;
4. les fichiers temporaires sont supprimés ;
5. l'API renvoie le texte complet, la langue utilisée et les segments horodatés.

Le service utilise un seul worker Uvicorn et n'accepte qu'une transcription à la
fois. Une requête concurrente reçoit une réponse HTTP `409`.

### Composants principaux

| Composant | Rôle |
| --- | --- |
| Python 3.12 | Environnement d'exécution |
| FastAPI, Pydantic et python-multipart | API HTTP, validation et réception des fichiers |
| Uvicorn | Serveur ASGI |
| OpenAI Whisper | Reconnaissance vocale et segmentation |
| PyTorch et CUDA | Inférence sur GPU, avec repli sur le CPU |
| FFmpeg | Décodage et conversion des formats audio |
| uv et `uv.lock` | Installation rapide et reproductible des dépendances Python |

La source de Whisper est figée sur un commit Git précis dans `pyproject.toml` afin
de rendre les installations reproductibles.

### Configuration

| Variable | Valeur dans le conteneur | Description |
| --- | --- | --- |
| `ASR_MODEL_NAME` | `base` | Nom du modèle Whisper à charger |
| `ASR_MODEL_PATH` | `/models` | Répertoire de téléchargement et de cache des modèles |

Le choix d'un modèle plus volumineux augmente la consommation de mémoire GPU. Au
premier démarrage, Whisper télécharge le modèle demandé : la machine doit donc
avoir accès à Internet et le démarrage peut prendre quelques minutes.

### Exécution locale

Python 3.12, [uv](https://docs.astral.sh/uv/) et FFmpeg doivent être installés sur
la machine :

```bash
uv sync --locked
ASR_MODEL_NAME=base ASR_MODEL_PATH=./models \
  uv run uvicorn run:app --host 127.0.0.1 --port 5002 --workers 1
```

Sans GPU CUDA accessible, cette commande fonctionne sur le CPU mais la
transcription est sensiblement plus lente.

### API

| Méthode | Route | Description |
| --- | --- | --- |
| `POST` | `/BatchTranscriptionService` | Transcrit le fichier envoyé dans le champ `audioFile` |
| `GET` | `/busy` | Indique si une transcription est en cours |
| `GET` | `/docs` | Ouvre la documentation interactive générée par FastAPI |

Exemple de transcription :

```bash
curl --fail --show-error \
  --form "audioFile=@./audio.flac" \
  http://localhost:5002/BatchTranscriptionService
```

La réponse suit cette structure :

```json
{
  "full_text": "Texte transcrit",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 2.5,
      "text": "Texte transcrit"
    }
  ],
  "language": "fr"
}
```

## Conteneurisation

### Préparer Docker pour le GPU

Le lancement GPU nécessite :

- un GPU NVIDIA compatible ;
- un pilote NVIDIA fonctionnel sur l'hôte (`nvidia-smi`) ;
- Docker Engine ;
- NVIDIA Container Toolkit.

Suivre le [guide officiel d'installation de NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
puis configurer le daemon Docker :

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

La première commande met à jour `/etc/docker/daemon.json` pour déclarer le runtime
NVIDIA. Une installation Docker en mode rootless nécessite une configuration
différente, également décrite dans le guide officiel.

Valider ensuite l'accès au GPU avec la
[charge de test officielle NVIDIA](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/sample-workload.html) :

```bash
sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

La [documentation Docker sur l'accès GPU](https://docs.docker.com/engine/containers/gpu/)
détaille également la sélection d'un ou de plusieurs GPU avec l'option `--gpus`.

### Construire l'image

Depuis ce répertoire :

```bash
docker build --tag transcribe-ai-mono-voix-stt:latest .
```

Le premier build télécharge notamment PyTorch et ses bibliothèques CUDA. Les
builds suivants réutilisent le cache BuildKit de `uv` tant que `pyproject.toml` et
`uv.lock` ne changent pas.

### Lancer le service sur le GPU

Créer d'abord un volume nommé pour conserver le modèle entre deux conteneurs :

```bash
docker volume create transcribe-ai-models
```

Puis lancer le service :

```bash
docker run --detach \
  --name transcribe-ai-mono-voix-stt \
  --restart unless-stopped \
  --gpus all \
  --publish 127.0.0.1:5002:5001 \
  --mount type=volume,source=transcribe-ai-models,target=/models \
  --env ASR_MODEL_NAME=base \
  transcribe-ai-mono-voix-stt:latest
```

Le port `5001` du conteneur est ainsi accessible sur
`http://localhost:5002`. L'adresse `127.0.0.1` limite volontairement l'accès à la
machine locale. Pour accepter des connexions externes, remplacer le mapping par
`--publish 5002:5001` et protéger l'API avec un pare-feu, une authentification ou
un reverse proxy.

### Vérifier le service

Suivre le chargement du modèle :

```bash
docker logs --follow transcribe-ai-mono-voix-stt
```

Vérifier depuis Python que PyTorch utilise CUDA :

```bash
docker exec transcribe-ai-mono-voix-stt python -c \
  "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Tester l'état et une transcription réelle :

```bash
curl --fail http://localhost:5002/busy

curl --fail --show-error \
  --form "audioFile=@./audio.flac" \
  http://localhost:5002/BatchTranscriptionService
```

Une réponse HTTP `200` contenant `full_text` et `segments`, combinée à
`CUDA: True`, valide le chemin complet d'inférence GPU.

Pour arrêter et supprimer le conteneur sans effacer le modèle mis en cache :

```bash
docker stop transcribe-ai-mono-voix-stt
docker rm transcribe-ai-mono-voix-stt
```

### Limites et sécurité

- La langue de transcription est actuellement fixée à `fr` dans le code.
- Une seule transcription est traitée à la fois.
- L'API n'implémente ni authentification ni limite explicite de taille d'upload ;
  elle ne doit pas être exposée directement sur Internet.
- Le volume monté sur `/models` doit rester accessible à l'utilisateur `10001` si
  un bind mount est utilisé à la place du volume Docker nommé.
- Triton devra être réintroduit si une future route active les timestamps mot à
  mot ou une fonctionnalité qui en dépend.
