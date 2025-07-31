# SocialAnalyzer 

**Automatisation intelligente de l'analyse qualitative d'entretiens sociologiques selon la méthode de Braun & Clarke (2006).**

Ce projet propose un pipeline automatisé qui exploite les modèles d'intelligence artificielle (**LLMs via Groq**) et **LangGraph** pour effectuer une analyse qualitative sociologique approfondie sur des corpus d'entretiens. L'analyse est réalisée selon la méthode reconnue de l’analyse thématique de **Braun & Clarke (2006)**.

---

## 🚀 Fonctionnalités clés

- **Extraction automatique des données** depuis des entretiens PDF.
- **Analyse thématique inductive** selon Braun & Clarke.
- **Génération automatique de rapports qualitatifs**, complets, structurés et interprétatifs.
- **Interface utilisateur intuitive** (Streamlit, LangGraph Studio) pour visualiser les résultats.
- **Workflow modulaire et personnalisable** grâce à LangGraph.

---

## 🎯 Objectifs du projet

- Automatiser l’analyse qualitative d'entretiens sociologiques.
- Produire rapidement des rapports interprétatifs et approfondis.
- Respecter les standards méthodologiques de Braun & Clarke.
- Simplifier, accélérer et améliorer la qualité des analyses sociologiques.

---

## ⚙️ Installation & Configuration

### 1. Clonez le dépôt

```bash
git clone https://github.com/votre_repo/my_langgraph_project.git
cd my_langgraph_project
```

### 2. Installez les dépendances

```bash
pip install -r requirements.txt
```


### 3. Créez un environnement virtuel
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.\.venv\Scripts\activate   # Windows
```
### 4. Configurez vos variables d'environnement
```bash
cp .env.example .env
```
GROQ_API_KEY=votre_cle_groq


### 5. Lancez le pipeline
```bash
python -m src.main --pdf data/raw/entretien1.pdf
python -m src.main --pdf data/raw/
python -m src.main --pdf data/raw/ --max_themes 5
```

### Interface utilisateur (LangGraph Studio & Streamlit)
```bash
langgraph dev # LangGraph Studio permet d’explorer et visualiser votre workflow. Lance le serveur LangGraph Studio
streamlit run app.py # Streamlit fournit une interface intuitive pour explorer les résultats
```

### 6. Méthodologie : Braun & Clarke (2006)

L'analyse qualitative suit rigoureusement la méthode thématique inductive proposée par Braun & Clarke :

1. Familiarisation avec les données

2. Génération de codes initiaux

3. Recherche des thèmes

4. Révision des thèmes

5. Définition et nommage des thèmes

6. Production du rapport interprétatif


### 7. Résultats et rapports
Les rapports générés sont sauvegardés dans :

- data/repport/
