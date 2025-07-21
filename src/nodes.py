import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import io # Ajouté pour le téléchargement
import tempfile # Ajouté pour les fichiers temporaires

import pandas as pd
import PyPDF2
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from groq import RateLimitError
# Retry configuration
GROQ_MODEL = "llama3-70b-8192"  # Ou 'llama3-8b-8192' pour plus rapide
TEMPERATURE = 0.7  # Créativité
MAX_RETRIES = 5
INITIAL_DELAY = 1

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --------------------------------------
# Configuration & Client Initialization
# --------------------------------------

def load_api_key() -> str:
    """Charge la clé API GROQ depuis .env."""
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("La variable d'environnement 'GROQ_API_KEY' n'est pas définie")
        raise ValueError("La variable d'environnement 'GROQ_API_KEY' n'est pas définie")
    logger.info("Clé API GROQ chargée.")
    return api_key

def create_groq_client(api_key: str) -> Groq:
    """Crée et retourne un client Groq."""
    return Groq(api_key=api_key)

# ----------------
# Extraction Texte (INCHANGÉ)
# ----------------

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extrait le texte d'un fichier PDF."""
    if not os.path.exists(pdf_path):
        logger.error(f"Le fichier temporaire {pdf_path} n'existe pas ou n'est pas accessible.")
        raise FileNotFoundError(f"Le fichier {pdf_path} n'existe pas")

    text = []
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            num_pages = len(reader.pages)
            logger.info(f"Extraction de texte depuis {pdf_path} ({num_pages} pages)")
            for i, page in enumerate(reader.pages):
                page_txt = page.extract_text() or ""
                text.append(page_txt)
                if (i + 1) % 10 == 0 or (i + 1) == num_pages: # Log progress
                     logger.info(f"Page {i+1}/{num_pages} extraite de {os.path.basename(pdf_path)}")
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction du PDF {pdf_path}: {e}", exc_info=True)
        raise

    logger.info(f"Extraction de texte terminée pour {pdf_path}")
    return "\n".join(text)


def clean_text(text: str) -> str:
    """Nettoie le texte en supprimant les espaces multiples."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ----------------------
# Étape 1 : Segmentation (INCHANGÉ)
# ----------------------

def segment_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> List[str]:
    """
    Découpe le texte en segments d'environ chunk_size mots avec chevauchement,
    en tentant de préserver les limites de phrases.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    segments = []
    current_segment = []
    current_size = 0

    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_size = len(sentence_words)

        if not sentence_words: # Skip empty sentences
            continue

        # Check if adding the sentence exceeds the chunk size significantly
        if current_size > 0 and current_size + sentence_size > chunk_size:
            # Add the current segment
            segments.append(" ".join(current_segment))

            # Determine overlap words based on the overlap size or sentence boundaries
            overlap_word_count = 0
            words_to_keep = []
            temp_segment_words = " ".join(current_segment).split()
            # Go back word by word until overlap size is reached or start is hit
            for i in range(len(temp_segment_words) - 1, -1, -1):
                 words_to_keep.insert(0, temp_segment_words[i])
                 overlap_word_count +=1
                 if overlap_word_count >= overlap:
                      # Check if we are in the middle of a sentence based on last word
                      if not temp_segment_words[i].endswith(('.', '!', '?')):
                          # Try to extend to the beginning of the sentence
                          potential_start = i - 1
                          while potential_start >= 0 and not temp_segment_words[potential_start].endswith(('.', '!', '?')):
                              words_to_keep.insert(0, temp_segment_words[potential_start])
                              potential_start -= 1
                      break # Stop once overlap is sufficient

            # Start new segment with overlap and the current sentence
            current_segment = words_to_keep + sentence_words
            current_size = len(current_segment)

        else:
            # Add sentence to the current segment
            current_segment.extend(sentence_words)
            current_size += sentence_size

    # Add the last segment if it exists
    if current_segment:
        segments.append(" ".join(current_segment))

    logger.info(f"Texte segmenté en {len(segments)} segments")
    # Log first few words of each segment for verification
    for i, seg in enumerate(segments[:3]):
        logger.debug(f"Segment {i+1} début: {' '.join(seg.split()[:10])}...")
    if len(segments) > 3:
         logger.debug("...")
         seg = segments[-1]
         logger.debug(f"Segment {len(segments)} début: {' '.join(seg.split()[:10])}...")

    return segments


# -------------------------------------
# Étape 2 & 3 : Analyse CoT + Codification (INCHANGÉ)
# -------------------------------------

def extract_json_from_text(text: str) -> Optional[List[Dict[str, Any]]]:
    """Extrait du JSON valide depuis un texte potentiellement mélangé."""
    # Essayer d'abord avec une recherche stricte de tableau JSON
    json_array_match = re.search(r"^\s*(\[.*\])\s*$", text, re.DOTALL)
    if json_array_match:
        json_str = json_array_match.group(1)
        try:
            # Valider avec le module json
            parsed = json.loads(json_str)
            if isinstance(parsed, list): # Assurer que c'est une liste
                logger.debug("JSON array extrait avec regex stricte.")
                return parsed
        except json.JSONDecodeError:
             logger.warning(f"Regex a trouvé un pattern JSON array, mais le parsing a échoué: {json_str[:100]}...")
             pass # Continuer avec d'autres méthodes

    # Rechercher un pattern JSON (array) dans le texte, même s'il y a du texte autour
    json_pattern = r"\[\s*\{.*?\}\s*(?:,\s*\{.*?\})*\s*\]" # Recherche un array d'objets
    match = re.search(json_pattern, text, re.DOTALL)

    if match:
        json_str = match.group(0)
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, list):
                logger.debug("JSON array extrait avec regex large.")
                return parsed
        except json.JSONDecodeError:
            logger.warning(f"Regex a trouvé un pattern JSON array potentiel, mais le parsing a échoué: {json_str[:100]}...")
            pass

    # Essayer d'extraire entre les premiers '[' et les derniers ']' comme fallback
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        potential_json = text[start : end + 1]
        try:
            parsed = json.loads(potential_json)
            # Vérifier si c'est une liste d'objets (ou une liste vide)
            if isinstance(parsed, list):
                 is_list_of_dicts = all(isinstance(item, dict) for item in parsed)
                 if is_list_of_dicts or not parsed: # Accepter liste vide ou liste de dicts
                    logger.debug("JSON array extrait entre crochets [ ... ].")
                    return parsed
                 else:
                     logger.warning("JSON extrait entre crochets n'est pas une liste de dictionnaires.")
            else:
                 logger.warning("JSON extrait entre crochets n'est pas une liste.")

        except json.JSONDecodeError:
            logger.warning(f"Tentative d'extraction JSON entre crochets [ ... ] a échoué pour: {potential_json[:100]}...")
            pass

    logger.warning("Aucun JSON valide (array de dicts) n'a pu être extrait du texte.")
    return None


def analyze_and_code_segment(
    client: Groq, model: str, segment: str, max_retries: int = 2
) -> List[Dict[str, Any]]:
    """
    Effectue une analyse CoT puis propose jusqu'à 3 codes (JSON).
    Inclut un mécanisme de réessai en cas d'échec.
    """
    # Construction du prompt avec exemples few-shot et instructions pour extraits longs
    prompt = (
        "Tu es un sociologue du numérique spécialisé en analyse qualitative.\n\n"
        "Étape 1: Analyse ce segment d'entretien en pensant à voix haute pour identifier les idées principales, "
        "y compris les idées implicites, les nuances ou l'ironie. (Ne pas inclure cette pensée dans la réponse finale JSON)\n\n"
        "Étape 2: Propose jusqu'à 3 codes sociologiques brefs (1-5 mots) accompagnés d'un extrait pertinent. "
        "Chaque code doit refléter un concept sociologique significatif.\n\n"
        "IMPORTANT: Pour chaque code, tu dois sélectionner un extrait LONG (au moins 15-20 mots OU une phrase complète significative) "
        "qui illustre bien le concept. Évite absolument les extraits trop courts (moins de 10 mots). "
        "L'extrait doit inclure suffisamment de contexte pour être compréhensible seul.\n\n"
        "Voici des exemples de thèmes et codes appropriés avec des extraits LONGS:\n\n"
        "EXEMPLE 1:\n"
        "Nom du thème: Un usage stratégique et pédagogique de l'IA générative\n"
        "Définition: Ce thème désigne l'intégration réfléchie de l'IA générative, en particulier de ChatGPT, dans les pratiques "
        "académiques des étudiant·es comme outil d'aide à la rédaction, à la compréhension et à la structuration des idées. "
        "L'usage est perçu comme complémentaire au travail intellectuel, mobilisé de manière ciblée pour dépasser des obstacles.\n"
        "Exemple verbatim: « le raisonnement que je vais présenter dans un travail, je vais plutôt le faire moi-même et j'ai plutôt "
        "utilisé ChatGpt pour m'aider pour des choses externes. Genre des biographies, des lectures, tout ça. Parce que je trouve "
        "que le raisonnement il n'est pas assez fort. »\n"
        'Codes possibles: [{ "code": "Complémentarité cognitive", "excerpt": "le raisonnement que je vais présenter dans un travail, je vais plutôt le faire moi-même et j\'ai plutôt utilisé ChatGpt pour m\'aider pour des choses externes" }, '
        '{ "code": "Usage ciblé et délimité", "excerpt": "utilisé ChatGpt pour m\'aider pour des choses externes. Genre des biographies, des lectures, tout ça. Parce que je trouve que le raisonnement il n\'est pas assez fort" }]\n\n'
        "EXEMPLE 2:\n"
        "Nom du thème: Un rapport critique à la légitimité académique menant à des tensions éthiques\n"
        "Définition: Ce thème renvoie à la manière dont les étudiant·es naviguent dans un espace institutionnel marqué par l'incertitude "
        "normative et le flou des règles entourant l'usage de l'IA générative. Entre méfiance vis-à-vis du plagiat et conscience des risques "
        "de transgression, les individus développent des stratégies d'auto-régulation.\n"
        "Exemple verbatim: « ça me brise le cœur de me dire, « bon ce mot est trop beau, il faut que je le mette quelque chose de plus humain, quoi. » »\n"
        'Codes possibles: [{ "code": "Anxiété normative", "excerpt": "ça me brise le cœur de me dire, « bon ce mot est trop beau, il faut que je le mette quelque chose de plus humain, quoi. »" }, '
        '{ "code": "Camouflage des traces IA", "excerpt": "ce mot est trop beau, il faut que je mette quelque chose de plus humain, quoi. » Pour que ça fasse plus comme si c\'était moi qui l\'avais écrit" }]\n\n'
        "EXEMPLE 3:\n"
        "Nom du thème: Une transformation du rapport au savoir et à l'écriture\n"
        "Définition: Ce thème interroge les effets de l'IA générative sur les manières de concevoir, produire et s'approprier le savoir dans "
        "le cadre universitaire. L'écriture n'est plus seulement une performance solitaire mais devient un processus dialogique, co-construit "
        "avec la machine.\n"
        "Exemple verbatim: « ça change les standards. Avant, on devait tout faire nous-mêmes, du coup au bout d'un moment t'as fait un travail "
        "pendant 3 semaines, au bout d'un moment t'as envie de le donner, même si c'est pas parfait, les mots utilisés sont pas parfaits, etc. "
        "Et là, vu qu'on a accès à un contenu de qualité relativement vite, je pense qu'il y a des standards où je me dis que ce n'est pas "
        "exactement ce que je veux. »\n"
        'Codes possibles: [{ "code": "Évolution des standards", "excerpt": "ça change les standards. Avant, on devait tout faire nous-mêmes, du coup au bout d\'un moment t\'as fait un travail pendant 3 semaines" }, '
        '{ "code": "Exigence accrue", "excerpt": "ça rend le truc plus exigeant parce qu\'on a accès à un contenu de qualité relativement vite" }, '
        '{ "code": "Temporalité transformée", "excerpt": "on a accès à un contenu de qualité relativement vite, je pense qu\'il y a des standards où je me dis que ce n\'est pas exactement ce que je veux" }]\n\n'
        f"Segment à analyser:\n'''\n{segment}\n'''\n\n"
        "Réponds UNIQUEMENT en format JSON avec un tableau (liste) contenant les codes et extraits, suivant ce modèle précis:\n"
        '[{"code": "concept sociologique", "excerpt": "extrait pertinent d\'au moins 15-20 mots ou une phrase complète"}, ...]\n'
        "Ne fournis AUCUN texte avant ou après le tableau JSON."
    )

    messages = [{"role": "user", "content": prompt}]
    segment_id_for_log = f"Segment starting with: '{segment[:50]}...'" # For logging

    for attempt in range(max_retries + 1):
        logger.info(f"Analyse CoT - Tentative {attempt+1}/{max_retries+1} pour {segment_id_for_log}")
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                max_tokens=1024, # Augmenté pour JSON + potentielle pensée CoT interne
                stream=False,
            )
            # Utiliser .strip() pour enlever les espaces/lignes vides avant/après
            raw_response_text = resp.choices[0].message.content.strip()
            logger.debug(f"Réponse brute reçue (tentative {attempt+1}):\n{raw_response_text}\n---")


            # Tentative d'extraction du JSON
            codes = extract_json_from_text(raw_response_text)

            if codes is not None: # Accepte une liste vide si le LLM n'a rien trouvé
                 # Vérification de la longueur des extraits
                valid_codes = []
                issues_found = False
                if not codes: # Si la liste est vide
                    logger.info(f"Analyse CoT: Le modèle n'a retourné aucun code pour {segment_id_for_log}. Accepté.")
                    return []

                for i, code in enumerate(codes):
                    if not isinstance(code, dict):
                         logger.warning(f"Item {i} dans la liste JSON n'est pas un dictionnaire: {code}")
                         issues_found = True
                         continue # Ignorer cet item
                    code_text = code.get("code")
                    excerpt = code.get("excerpt")

                    if not code_text or not excerpt:
                        logger.warning(f"Code JSON incomplet trouvé (manque 'code' ou 'excerpt'): {code}")
                        issues_found = True
                        continue # Ignorer ce code

                    excerpt_words = excerpt.split()
                    if len(excerpt_words) >= 10:
                        valid_codes.append(code)
                    else:
                        logger.warning(f"Extrait trop court détecté (< 10 mots) pour le code '{code_text}': '{excerpt}'. Tentative de correction échouée ou non tentée par le modèle.")
                        # On ne tente pas de l'étendre ici, on le signale juste. Le Juge s'en chargera.
                        # Ou on pourrait décider de l'ignorer ici, mais le garder pour le Juge est peut-être mieux.
                        valid_codes.append(code) # Garder même si court, pour que le Juge le voie.
                        issues_found = True # Marquer qu'il y a eu un problème potentiel

                # Si on a extrait quelque chose et qu'il n'y a pas eu d'erreur majeure de format
                # On retourne les codes (même ceux potentiellement courts, le Juge filtrera)
                if valid_codes or not issues_found: # Si on a des codes valides OU si on a extrait une liste vide sans erreur
                     logger.info(f"Analyse CoT réussie (tentative {attempt+1}). {len(valid_codes)} codes extraits pour {segment_id_for_log}.")
                     return valid_codes
                # Si on a eu des problèmes (codes incomplets, etc.) mais qu'on a quand même extrait *quelque chose*,
                # on passe à la tentative suivante si possible.
                elif attempt < max_retries:
                     logger.warning(f"Problèmes détectés dans le JSON (codes incomplets ou extraits courts signalés). Tentative {attempt+2} demandée.")
                     # Préparer le message pour la nouvelle tentative
                     fix_prompt = (
                         "Ta réponse précédente n'était pas entièrement satisfaisante (peut-être JSON invalide, incomplet, ou extraits trop courts).\n"
                         "Assure-toi de fournir UNIQUEMENT un tableau JSON valide contenant des objets avec les clés 'code' et 'excerpt'.\n"
                         "Les EXTRAITS doivent être LONGS (minimum 15-20 mots ou une phrase complète).\n"
                         'Format attendu: [{"code": "...", "excerpt": "extrait long"}, ...]'
                     )
                     messages = [
                         {"role": "user", "content": prompt}, # Rappeler le contexte initial
                         {"role": "assistant", "content": raw_response_text}, # Montrer la réponse précédente
                         {"role": "user", "content": fix_prompt}, # Demander la correction
                     ]
                # else: on a eu des soucis et c'était la dernière tentative

            # Si extract_json_from_text retourne None OU si on arrive ici après échec de validation
            elif attempt < max_retries:
                logger.warning(f"Échec de l'extraction JSON ou JSON invalide (tentative {attempt+1}). Tentative {attempt+2} demandée.")
                fix_prompt = (
                    "Ta réponse précédente n'a pas pu être interprétée comme un JSON valide ou était incorrecte.\n"
                    "Réponds avec un tableau JSON valide au format:\n"
                    '[{"code": "concept sociologique", "excerpt": "extrait long et pertinent"}, ...]\n'
                    "N'inclus aucune explication ou texte en dehors du JSON.\n"
                    "Assure-toi que les extraits soient longs (15-20 mots min)."
                )
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": raw_response_text},
                    {"role": "user", "content": fix_prompt},
                ]
            # else: c'était la dernière tentative et extract_json a retourné None

        except Exception as e:
            logger.error(
                f"Erreur inattendue lors de l'analyse du segment (tentative {attempt+1}) pour {segment_id_for_log}: {e}", exc_info=True
            )
            if attempt == max_retries:
                logger.error(f"Échec final de l'analyse pour {segment_id_for_log} après plusieurs tentatives.")
                return [] # Retourner une liste vide en cas d'échec final

    # Si on sort de la boucle sans succès
    logger.error(
        f"Impossible d'obtenir un JSON valide après {max_retries + 1} tentatives pour {segment_id_for_log}. Retour d'une liste vide."
    )
    return []


# -----------------------------
# Étape 4 : Validation (juge) (INCHANGÉ)
# -----------------------------

def judge_codes(
    client: Groq, model: str, segment: str, codes: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Valide ou corrige chaque code proposé par l'agent.
    Retourne une liste de codes validés/améliorés.
    S'assure que les extraits sont suffisamment longs (au moins 10 mots).
    """
    if not codes:
        return []
    
    excerpts = [c["excerpt"] for c in codes if "excerpt" in c]
    reduced_segment = extract_relevant_sentences(segment, excerpts)

    segment_id_for_log = f"Segment starting with: '{segment[:50]}...'" # For logging
    logger.info(f"Validation Juge: Démarrage pour {len(codes)} codes de {segment_id_for_log}.")

    # Convertir les codes reçus en JSON pour le prompt
    try:
        codes_json = json.dumps(codes, ensure_ascii=False, indent=2)
    except TypeError:
        logger.error("Erreur lors de la conversion des codes en JSON pour le prompt du juge.")
        return codes # Retourner les codes originaux en cas d'erreur de sérialisation



    # Construction du prompt avec exemples few-shot pour la validation et insistance sur extraits longs
    prompt = (
        "Tu es un expert en sociologie évaluant des codes d'analyse qualitative.\n\n"
        f"Voici un segment d'entretien:\n'''\n{reduced_segment}\n'''\n\n"
        f"Et voici les codes proposés:\n{codes_json}\n\n"
        "CRITÈRE IMPORTANT: Pour chaque code :\n"
        "1. Évalue la pertinence sociologique du 'code'.\n"
        "2. Vérifie que l'extrait ('excerpt') est LONG (minimum 10 mots, idéalement 15-20 ou une phrase complète) et illustre bien le code.\n"
        "3. Si un extrait est trop court ou mal choisi, tu DOIS l'étendre ou le corriger en cherchant dans le segment original pour fournir un meilleur contexte.\n"
        "4. Si un code n'est pas pertinent ou ne peut être justifié par un extrait approprié, supprime-le.\n\n"
        "Voici des exemples de bons codes sociologiques et de leurs extraits pertinents LONGS pour t'inspirer:\n\n"
        # ... (les exemples restent les mêmes) ...
        "EXEMPLE 1 - Concernant l'usage de l'IA dans les travaux académiques:\n"
        '- Code: "Complémentarité cognitive" avec extrait: "le raisonnement que je vais présenter dans un travail, je vais plutôt le faire moi-même et j\'ai plutôt utilisé ChatGpt pour m\'aider pour des choses externes"\n'
        '- Code: "Usage ciblé et délimité" avec extrait: "utilisé ChatGpt pour m\'aider pour des choses externes. Genre des biographies, des lectures, tout ça. Parce que je trouve que le raisonnement il n\'est pas assez fort"\n\n'
        "EXEMPLE 2 - Concernant les tensions éthiques liées à l'IA:\n"
        '- Code: "Anxiété normative" avec extrait: "ça me brise le cœur de me dire, « bon ce mot est trop beau, il faut que je le mette quelque chose de plus humain, quoi. » Pour que ça fasse plus comme si c\'était moi qui l\'avais écrit"\n'
        '- Code: "Camouflage des traces IA" avec extrait: "ce mot est trop beau, il faut que je mette quelque chose de plus humain, quoi. » Pour que ça fasse plus comme si c\'était moi qui l\'avais écrit et pas ChatGPT"\n\n'

        '[{"code": "code validé ou amélioré", "excerpt": "extrait pertinent LONG (min 10 mots)"}, ...]\n\n'
        "Assure-toi que chaque objet dans le tableau a bien les clés 'code' et 'excerpt' et que les extraits respectent la longueur minimale. Ne retourne RIEN d'autre que le JSON."
    )

    max_retries_judge = 1 # Moins de tentatives pour le juge, il doit être plus direct
    for attempt in range(max_retries_judge + 1):
        logger.info(f"Validation Juge - Tentative {attempt+1}/{max_retries_judge+1} pour {segment_id_for_log}")
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2048,  # Augmenté pour permettre des extraits longs et potentiellement plus de codes
                stream=False,
            )
            content = resp.choices[0].message.content.strip()
            logger.debug(f"Réponse brute reçue du Juge (tentative {attempt+1}):\n{content}\n---")

            # Extraction du JSON de la réponse du juge
            validated_codes = extract_json_from_text(content)

            if validated_codes is not None: # Accepte une liste vide si le juge supprime tout
                final_codes = []
                issues_found = False
                if not validated_codes:
                     logger.info(f"Validation Juge: Le juge a supprimé tous les codes pour {segment_id_for_log}.")
                     return []

                for i, code in enumerate(validated_codes):
                     if not isinstance(code, dict):
                         logger.warning(f"Item {i} du Juge n'est pas un dictionnaire: {code}")
                         issues_found = True
                         continue
                     code_text = code.get("code")
                     excerpt = code.get("excerpt")

                     if not code_text or not excerpt:
                         logger.warning(f"Code du Juge incomplet trouvé (manque 'code' ou 'excerpt'): {code}")
                         issues_found = True
                         continue

                     excerpt_words = excerpt.split()
                     if len(excerpt_words) >= 10:
                         final_codes.append(code)
                     else:
                         # Le juge n'a pas respecté la consigne de longueur minimale !
                         logger.warning(f"Juge a retourné un extrait trop court (< 10 mots) pour code '{code_text}': '{excerpt}'. On essaie de l'étendre manuellement.")
                         issues_found = True
                         # Tentative manuelle d'extension si l'extrait court est dans le segment
                         extended_excerpt = None
                         if excerpt in segment:
                              # Essayer de trouver la phrase complète contenant l'extrait court
                              # Split par ponctuations de fin de phrase, en conservant les délimiteurs
                              sentences = re.split(r'([.!?])\s*', segment)
                              # Reconstruire les phrases avec leur ponctuation
                              full_sentences = []
                              if sentences:
                                   current_sentence = sentences[0]
                                   for j in range(1, len(sentences), 2):
                                        if j+1 < len(sentences):
                                             current_sentence += (sentences[j] or '') + (sentences[j+1] or '')
                                             full_sentences.append(current_sentence.strip())
                                             current_sentence = "" # Start new sentence conceptually
                                        else: # Handle last part if no trailing punctuation
                                             current_sentence += (sentences[j] or '')
                                             full_sentences.append(current_sentence.strip())
                                   # Add the last part if not empty and not added
                                   if current_sentence and current_sentence.strip() and current_sentence.strip() not in full_sentences:
                                       full_sentences.append(current_sentence.strip())


                              for sentence in full_sentences:
                                   if excerpt in sentence and len(sentence.split()) >= 10:
                                        extended_excerpt = sentence
                                        logger.info(f"Extension manuelle réussie pour l'extrait court: '{extended_excerpt}'")
                                        break
                         if extended_excerpt:
                             code["excerpt"] = extended_excerpt # Mettre à jour l'extrait dans le dictionnaire
                             final_codes.append(code)
                         else:
                             logger.warning(f"Impossible d'étendre manuellement l'extrait court fourni par le juge pour '{code_text}'. Ce code sera ignoré.")
                             # Ne pas ajouter ce code si l'extension échoue

                # Si on a extrait quelque chose et qu'il n'y a pas eu d'erreur majeure de format, OU si on a une liste vide valide
                if final_codes or not issues_found:
                     logger.info(f"Validation Juge réussie (tentative {attempt+1}). {len(final_codes)} codes validés/améliorés pour {segment_id_for_log}.")
                     return final_codes
                # Si on a eu des problèmes et qu'on a encore des tentatives
                elif attempt < max_retries_judge:
                     logger.warning(f"Problèmes détectés dans la réponse du Juge (codes incomplets, extraits trop courts non corrigés). Nouvelle tentative demandée.")
                     # Le prompt de retry pourrait être similaire à celui de l'analyseur
                     fix_prompt = (
                        "Ta réponse précédente contenait des erreurs (JSON invalide, codes incomplets, ou extraits encore trop courts).\n"
                        "Corrige ta réponse pour fournir UNIQUEMENT un tableau JSON valide.\n"
                        "Chaque élément doit avoir 'code' et 'excerpt'.\n"
                        "Chaque 'excerpt' DOIT avoir au moins 10 mots (idéalement 15-20 ou phrase complète).\n"
                        'Format: [{"code": "...", "excerpt": "extrait LONG"}, ...]'
                     )
                     # Note: On ne renvoie pas la réponse erronée au juge pour éviter confusion, on réitère juste la demande.
                     # Ceci est différent de l'approche pour l'analyseur CoT.
                     messages = [{"role": "user", "content": prompt}] # On redonne le prompt initial complet
                # else: dernière tentative échouée

            # Si extract_json_from_text retourne None OU si on arrive ici après échec de validation/correction
            elif attempt < max_retries_judge:
                logger.warning(f"Échec de l'extraction JSON de la réponse du Juge (tentative {attempt+1}). Nouvelle tentative demandée.")
                messages = [{"role": "user", "content": prompt}] # On redonne le prompt initial complet

        except Exception as e:
            logger.error(f"Erreur inattendue lors de la validation Juge (tentative {attempt+1}) pour {segment_id_for_log}: {e}", exc_info=True)
            if attempt == max_retries_judge:
                 logger.error(f"Échec final de la validation Juge pour {segment_id_for_log}. Retour des codes originaux.")
                 return codes # Retourner les codes originaux en cas d'échec grave du juge

    # Si on sort de la boucle sans succès après les tentatives
    logger.error(f"Impossible d'obtenir une réponse valide du Juge après {max_retries_judge + 1} tentatives pour {segment_id_for_log}. Retour des codes originaux.")
    return codes # Fallback: retourner les codes originaux si le juge échoue complètement

def extract_relevant_sentences(segment: str, excerpts: List[str]) -> str:
    """Récupère les phrases contenant les excerpts pour réduire la taille du segment."""
    sentences = re.split(r'(?<=[.!?])\s+', segment)
    relevant = []
    for sent in sentences:
        for excerpt in excerpts:
            if excerpt.strip() in sent:
                relevant.append(sent.strip())
                break
    return "\n".join(relevant)
# --------------------------------
# Étape 5 : Clustering des codes (LIMITÉ À 10 THÈMES MAX) (INCHANGÉ)
# --------------------------------

def cluster_codes_limited(
    all_codes: List[Dict[str, Any]],
    max_clusters: int = 10,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[str, int]]:
    """
    Regroupe les codes par similarité sémantique en limitant à max_clusters thèmes.
    Retourne un dict cluster_id -> list de codes et un mapping code -> cluster_id.
    """
    if not all_codes:
        logger.warning("Clustering: Reçu une liste de codes vide. Retour de résultats vides.")
        return {}, {}

    # Extraire uniquement les textes des codes
    # Utiliser un set pour dédoublonner les textes de code avant l'embedding
    unique_code_texts = list(set(code["code"] for code in all_codes))
    code_text_to_index = {text: i for i, text in enumerate(unique_code_texts)}
    logger.info(f"Clustering: {len(all_codes)} codes reçus, {len(unique_code_texts)} textes de code uniques à traiter.")


    if not unique_code_texts:
        logger.warning("Clustering: Aucun texte de code unique trouvé après filtrage. Retour de résultats vides.")
        return {}, {}

    try:
        # Générer les embeddings pour les codes uniques
        logger.info(f"Génération des embeddings pour {len(unique_code_texts)} codes uniques avec '{embedding_model}'...")
        embedder = SentenceTransformer(embedding_model)
        embeddings = embedder.encode(unique_code_texts, show_progress_bar=True)
        logger.info("Embeddings générés.")

        # Déterminer le nombre de clusters à utiliser
        # On ne peut pas avoir plus de clusters que de points de données (codes uniques)
        n_clusters_effective = min(max_clusters, len(unique_code_texts))
        if n_clusters_effective < 1:
             logger.error("Clustering: Le nombre effectif de clusters est inférieur à 1. Impossible de clusteriser.")
             return {}, {} # Ou gérer ce cas différemment

        logger.info(f"Clustering hiérarchique demandé avec n_clusters = {n_clusters_effective} (max_clusters={max_clusters}, unique_codes={len(unique_code_texts)})")

        # Clustering hiérarchique agglomératif
        # 'ward' minimise la variance au sein de chaque cluster
        # 'cosine' est souvent bon pour les embeddings textuels, mais AgglomerativeClustering avec 'ward' utilise 'euclidean' par défaut.
        # Testons avec 'ward' et 'euclidean' (défaut). Si les résultats sont mauvais, 'average' et 'cosine' pourraient être une alternative.
        clustering = AgglomerativeClustering(n_clusters=n_clusters_effective, metric='euclidean', linkage='ward')
        # Les labels correspondent à l'ordre des unique_code_texts
        labels = clustering.fit_predict(embeddings)
        logger.info("Clustering terminé.")

        # Créer un mapping du texte de code unique vers son label de cluster
        code_text_to_cluster_label = {text: int(labels[i]) for i, text in enumerate(unique_code_texts)}

        # Organiser tous les codes originaux (y compris les doublons) en clusters
        clusters: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(n_clusters_effective)}
        code_to_cluster_map: Dict[str, int] = {} # Pour le retour final (mapping code -> cluster id)

        processed_code_texts_in_map = set() # Pour éviter les doublons dans code_to_cluster_map si le même code apparaît plusieurs fois

        for code_obj in all_codes:
            code_text = code_obj["code"]
            # Trouver le label de cluster pour ce texte de code
            cluster_label = code_text_to_cluster_label.get(code_text)
            if cluster_label is not None:
                clusters[cluster_label].append(code_obj)
                # Ajouter au mapping de retour seulement si pas déjà fait pour ce texte de code
                if code_text not in processed_code_texts_in_map:
                    code_to_cluster_map[code_text] = cluster_label
                    processed_code_texts_in_map.add(code_text)
            else:
                # Ne devrait pas arriver si tous les codes sont dans unique_code_texts
                 logger.warning(f"Code '{code_text}' non trouvé dans le mapping de clustering après traitement. Ignoré.")


        # Filtrer les clusters vides (peu probable avec Agglomerative Clustering mais par sécurité)
        final_clusters = {k: v for k, v in clusters.items() if v}
        num_final_clusters = len(final_clusters)

        logger.info(
            f"Clustering terminé: {num_final_clusters} thèmes (clusters non vides) identifiés (limité à {max_clusters})."
        )
        if num_final_clusters < n_clusters_effective:
             logger.warning(f"Moins de clusters non vides ({num_final_clusters}) que demandé ({n_clusters_effective}).")


        return final_clusters, code_to_cluster_map

    except ImportError as ie:
         logger.error(f"Erreur d'importation liée au clustering (SentenceTransformer ou Scikit-learn): {ie}. Assurez-vous que les bibliothèques sont installées.")
         raise
    except Exception as e:
        logger.error(f"Erreur grave lors du clustering: {e}", exc_info=True)
        # Fallback très simple: créer max_clusters groupes arbitraires si tout échoue
        logger.warning("Tentative de fallback de clustering simple.")
        simple_clusters = {}
        simple_code_to_cluster = {}
        if not all_codes: return {}, {}

        # Regroupement simplifié pour fallback
        num_codes = len(all_codes)
        codes_per_cluster = max(1, (num_codes + max_clusters - 1) // max_clusters) # Arrondi supérieur

        current_code_index = 0
        fallback_cluster_id = 0
        processed_codes_fallback = set() # Pour le map

        while current_code_index < num_codes and fallback_cluster_id < max_clusters:
            cluster_codes = all_codes[current_code_index : current_code_index + codes_per_cluster]
            if cluster_codes:
                simple_clusters[fallback_cluster_id] = cluster_codes
                for code in cluster_codes:
                    code_text = code['code']
                    if code_text not in processed_codes_fallback:
                         simple_code_to_cluster[code_text] = fallback_cluster_id
                         processed_codes_fallback.add(code_text)

                fallback_cluster_id += 1
                current_code_index += len(cluster_codes)
            else:
                 break # Should not happen if logic is right

        logger.warning(f"Fallback clustering créé avec {len(simple_clusters)} groupes.")
        return simple_clusters, simple_code_to_cluster


# -------------------------------
# Étape 6 : Labelling des thèmes (INCHANGÉ)
# -------------------------------

def label_themes(
    client: Groq, model: str, clusters: Dict[int, List[Dict[str, Any]]]
) -> Dict[int, str]:
    """
    Pour chaque cluster, demande au LLM un titre de thème sociologique concis.
    """
    theme_labels: Dict[int, str] = {}
    if not clusters:
        logger.warning("Labelling: Reçu un dictionnaire de clusters vide.")
        return {}

    logger.info(f"Démarrage du labelling pour {len(clusters)} clusters.")

    # Exemple few-shot pour le labelling
    examples = [
        "Usage stratégique IA", # Raccourci pour le prompt
        "Tensions éthiques IA",
        "Transformation savoir/écriture",
    ]

    # Limiter le nombre d'exemples dans le prompt pour économiser les tokens si beaucoup de clusters
    MAX_ITEMS_PER_CLUSTER_FOR_PROMPT = 5

    for cluster_id, codes in clusters.items():
        if not codes:
            logger.warning(f"Cluster {cluster_id} est vide, impossible de le labelliser.")
            continue

        # Extraire le texte des codes et quelques excerpts pour le contexte
        # Prendre un échantillon si le cluster est très grand
        sample_codes = codes[:MAX_ITEMS_PER_CLUSTER_FOR_PROMPT]
        # Créer une représentation textuelle simple pour le prompt
        code_items_repr = []
        for c in sample_codes:
             # Tronquer les extraits longs pour le prompt
             excerpt_preview = c.get('excerpt', '')
             if len(excerpt_preview.split()) > 25:
                 excerpt_preview = ' '.join(excerpt_preview.split()[:25]) + '...'
             code_items_repr.append(f"- Code: \"{c.get('code', 'N/A')}\" (Ex: \"{excerpt_preview}\")")

        codes_representation = "\n".join(code_items_repr)
        if len(codes) > MAX_ITEMS_PER_CLUSTER_FOR_PROMPT:
             codes_representation += f"\n... (et {len(codes) - MAX_ITEMS_PER_CLUSTER_FOR_PROMPT} autres codes)"

        prompt = (
            "Tu es un sociologue synthétisant des résultats d'analyse qualitative.\n\n"
            f"Voici un groupe de codes apparentés issus d'un cluster (Thème ID: {cluster_id}):\n{codes_representation}\n\n"
            "Propose un titre de thème sociologique TRÈS CONCIS (2-5 mots maximum) qui capture l'essence commune de ces codes.\n"
            "Le titre doit être pertinent par rapport aux codes et extraits fournis.\n\n"
            "Voici des exemples de BONS titres de thèmes CONCIS:\n"
            f'- "{examples[0]}"\n'
            f'- "{examples[1]}"\n'
            f'- "{examples[2]}"\n\n'
            'Réponds UNIQUEMENT en format JSON: {"theme": "Ton titre concis ici"}\n'
            "Ne fournis AUCUN autre texte."
        )

        theme_name = f"Thème non labellisé {cluster_id+1}" # Fallback name
        try:
            logger.info(f"Labelling du cluster {cluster_id} ({len(codes)} codes)...")
            resp = client.chat.completions.create(
                model=model, # Utiliser le même modèle que pour l'analyse CoT ou un modèle rapide ? Testons avec CoT.
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=128, # Devrait être suffisant pour {"theme": "..."}
                stream=False,
                # response_format={"type": "json_object"}, # Si le modèle supporte le mode JSON strict
            )
            content = resp.choices[0].message.content.strip()
            logger.debug(f"Réponse brute reçue pour labelling cluster {cluster_id}:\n{content}\n---")


            # Extraction du JSON - plus robuste
            match = re.search(r'\{\s*"theme"\s*:\s*"([^"]+)"\s*\}', content)
            if match:
                theme_name = match.group(1).strip()
                # Vérifier la concision (optionnel mais recommandé par le prompt)
                if len(theme_name.split()) > 6: # Permettre un mot de plus que demandé
                    logger.warning(f"Label pour cluster {cluster_id} est long: '{theme_name}'. Raccourcissement suggéré.")
                    # On pourrait essayer de le raccourcir ici ou juste l'accepter. Acceptons-le pour l'instant.
            else:
                # Tentative d'extraction JSON complète si regex échoue
                try:
                    data = json.loads(content)
                    if isinstance(data, dict) and "theme" in data and isinstance(data["theme"], str):
                        theme_name = data["theme"].strip()
                        logger.info(f"Label extrait via parsing JSON pour cluster {cluster_id}: '{theme_name}'")
                        if len(theme_name.split()) > 6:
                           logger.warning(f"Label pour cluster {cluster_id} est long: '{theme_name}'.")
                    else:
                        logger.warning(f"Réponse JSON pour cluster {cluster_id} n'a pas le format attendu {{'theme': '...'}}. Contenu: {content}")
                        # Essayer de prendre le contenu brut comme label si c'est une chaîne courte? Risqué. Utiliser fallback.
                except json.JSONDecodeError:
                    logger.warning(f"Réponse pour cluster {cluster_id} n'est pas un JSON valide ou n'a pas pu être extraite. Contenu: {content}")
                    # Essayer de récupérer le texte s'il est court et ressemble à un label?
                    if len(content.split()) <= 6 and not content.startswith("{") and not content.startswith("["):
                         theme_name = content
                         logger.info(f"Utilisation du texte brut comme label (fallback) pour cluster {cluster_id}: '{theme_name}'")
                    # Sinon, le fallback 'Thème non labellisé...' sera utilisé.


        except Exception as e:
            logger.error(f"Erreur lors du labelling du cluster {cluster_id}: {e}", exc_info=True)
            # Le fallback 'Thème non labellisé...' sera utilisé

        logger.info(f"Cluster {cluster_id} labellisé comme: '{theme_name}'")
        theme_labels[cluster_id] = theme_name

    logger.info("Labelling des thèmes terminé.")
    return theme_labels


# -----------------------------------
# Étape 7 : Second-level clustering (pour regrouper si > 10 thèmes) (INCHANGÉ)
# -----------------------------------

def meta_cluster_themes(
    client: Groq,
    model: str,
    clusters: Dict[int, List[Dict[str, Any]]],
    theme_map: Dict[int, str],
    target_count: int = 10,
) -> Tuple[Dict[int, str], Dict[int, int]]:
    """
    Si nécessaire, regroupe les thèmes en méta-thèmes pour atteindre le nombre cible.
    Retourne un dict meta_theme_id -> nom du méta-thème et un mapping original_cluster_id -> meta_theme_id.
    """
    num_initial_themes = len(clusters)
    if num_initial_themes <= target_count:
        logger.info(f"Méta-clustering non requis ({num_initial_themes} thèmes <= cible de {target_count}).")
        # Retourner les thèmes comme leurs propres méta-thèmes
        # Le mapping meta_theme_id -> nom
        meta_theme_names = {cluster_id: theme_map.get(cluster_id, f"Thème {cluster_id}")
                           for cluster_id in clusters.keys()}
        # Le mapping original_cluster_id -> meta_theme_id (qui est lui-même ici)
        theme_to_meta_assignment = {cluster_id: cluster_id for cluster_id in clusters.keys()}
        return meta_theme_names, theme_to_meta_assignment

    logger.info(f"Méta-clustering requis: {num_initial_themes} thèmes > cible de {target_count}. Démarrage du regroupement.")

    # Pour chaque thème, créer une description synthétique
    theme_summaries = {}
    MAX_CODES_PER_THEME_FOR_PROMPT = 5 # Limiter pour le prompt de méta-clustering
    for cluster_id, codes in clusters.items():
        theme_name = theme_map.get(cluster_id, f"Thème {cluster_id}")
        # Prendre les codes les plus fréquents ou les premiers N codes comme représentatifs
        representative_codes = [c['code'] for c in codes[:MAX_CODES_PER_THEME_FOR_PROMPT]]
        codes_str = ", ".join(f'"{code}"' for code in representative_codes)
        if len(codes) > MAX_CODES_PER_THEME_FOR_PROMPT:
            codes_str += ", ..."
        # Utiliser l'ID original du cluster comme clé dans le prompt
        theme_summaries[str(cluster_id)] = ( # Utiliser str(cluster_id) car JSON keys must be strings
            f"Thème ID {cluster_id} ('{theme_name}'): contient des codes comme [{codes_str}]"
        )

    # Demander au LLM de regrouper les thèmes avec exemples few-shot
    # Convertir les résumés en une chaîne formatée pour le prompt
    themes_description_str = "\n".join(f"- {desc}" for desc in theme_summaries.values())

    prompt = (
        "Tu es un expert en sociologie chargé de regrouper des thèmes d'analyse qualitative similaires.\n\n"
        f"Voici {num_initial_themes} thèmes identifiés (avec leurs ID et quelques codes exemples):\n{themes_description_str}\n\n"
        f"Regroupe ces thèmes en EXACTEMENT {target_count} méta-thèmes maximum. Chaque thème original (identifié par son ID) doit être assigné "
        f"à un seul méta-thème plus large.\n\n"
        "Pour chaque méta-thème créé, propose un nom concis (2-6 mots).\n\n"
        "Voici un exemple de regroupement thématique en sociologie du numérique:\n\n"
        '- Méta-thème "Pratiques numériques et apprentissage" pourrait inclure les thèmes originaux avec ID 3 ("Usage pédagogique des outils"), 7 ("Appropriation des technologies") et 12 ("Stratégies d\'adaptation numérique").\n'
        '- Méta-thème "Tensions éthiques et normatives" pourrait inclure les thèmes ID 1 ("Anxiété face aux règles"), 5 ("Légitimité académique contestée") et 9 ("Stratégies de contournement").\n'
        # ... (peut-être ajouter un 3ème exemple) ...
        "\nRéponds UNIQUEMENT en format JSON avec le format suivant:\n"
        '{\n'
        '  "meta_themes": {\n'
        '    "0": "Nom du Méta-thème 1",\n'
        '    "1": "Nom du Méta-thème 2",\n'
        f'    ...\n'
        f'    "{target_count-1}": "Nom du Méta-thème {target_count}"\n' # Assurer qu'il y a target_count clés
        '  },\n'
        '  "assignments": {\n'
        '    "original_theme_id_1": meta_theme_id_for_1, \n' # ex: "3": 0
        '    "original_theme_id_2": meta_theme_id_for_2, \n' # ex: "7": 0
        '    "original_theme_id_5": meta_theme_id_for_5, \n' # ex: "1": 1
        '    ...\n'
        '    "last_original_theme_id": its_meta_theme_id \n' # ex: "12": 0
        '  }\n'
        '}\n\n'
        "Où 'meta_themes' est un dictionnaire des méta-thèmes numérotés de 0 à {target_count-1}. "
        "Et 'assignments' indique pour chaque ID de thème original (clé en chaîne de caractères) l'ID numérique (0 à {target_count-1}) du méta-thème auquel il est assigné.\n"
        "Assure-toi que CHAQUE thème original listé ci-dessus a une entrée dans 'assignments' et que les ID de méta-thème assignés sont valides (entre 0 et {target_count-1})."
    )

    final_meta_theme_names: Dict[int, str] = {}
    final_theme_to_meta_assignment: Dict[int, int] = {}

    try:
        logger.info("Envoi de la requête de méta-clustering au LLM...")
        resp = client.chat.completions.create(
            model=model, # Utiliser le même modèle puissant
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, # Un peu plus de créativité pour le regroupement et nommage
            max_tokens=2048, # Augmenté car le prompt est long et la réponse peut l'être aussi
            stream=False,
            # response_format={"type": "json_object"}, # Si supporté
        )
        content = resp.choices[0].message.content.strip()
        logger.debug(f"Réponse brute reçue pour méta-clustering:\n{content}\n---")


        # Extraction du JSON de la réponse
        # Utiliser une regex pour extraire le bloc JSON principal, plus robuste aux commentaires
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                data = json.loads(json_str)
                raw_meta_themes = data.get("meta_themes")
                raw_assignments = data.get("assignments")

                if isinstance(raw_meta_themes, dict) and isinstance(raw_assignments, dict):
                    # Valider et convertir les meta_themes
                    valid_meta_themes = {}
                    all_meta_ids = set()
                    for k, v in raw_meta_themes.items():
                        try:
                            meta_id = int(k)
                            if 0 <= meta_id < target_count and isinstance(v, str) and v.strip():
                                valid_meta_themes[meta_id] = v.strip()
                                all_meta_ids.add(meta_id)
                            else:
                                logger.warning(f"Méta-thème invalide dans la réponse: key={k}, value={v}. Ignoré.")
                        except ValueError:
                             logger.warning(f"Clé de méta-thème non entière: {k}. Ignoré.")

                    # S'assurer qu'on a le bon nombre de méta-thèmes, sinon ajouter des placeholders
                    if len(valid_meta_themes) < target_count:
                         logger.warning(f"Le LLM a fourni {len(valid_meta_themes)} méta-thèmes, mais {target_count} étaient attendus. Ajout de placeholders.")
                         for i in range(target_count):
                              if i not in valid_meta_themes:
                                   valid_meta_themes[i] = f"Méta-thème {i+1} (Généré)"
                                   all_meta_ids.add(i)

                    # Valider et convertir les assignments
                    valid_assignments = {}
                    assigned_original_themes = set()
                    original_theme_ids_str = set(theme_summaries.keys()) # Les ID des thèmes qu'on a envoyés (en str)

                    for k, v in raw_assignments.items():
                         if k not in original_theme_ids_str:
                              logger.warning(f"Assignment reçu pour un ID de thème original inconnu: {k}. Ignoré.")
                              continue
                         try:
                             original_id = int(k) # Convertir la clé en int pour le mapping final
                             meta_id = int(v)
                             if meta_id in all_meta_ids: # Vérifier si l'ID de méta-thème assigné est valide
                                 valid_assignments[original_id] = meta_id
                                 assigned_original_themes.add(k)
                             else:
                                 logger.warning(f"Assignment invalide: thème original {k} assigné au méta-thème inexistant {v}. Ignoré.")
                         except (ValueError, TypeError):
                             logger.warning(f"Assignment invalide (non entier): key={k}, value={v}. Ignoré.")

                    # Vérifier si tous les thèmes originaux ont été assignés
                    missing_assignments = original_theme_ids_str - assigned_original_themes
                    if missing_assignments:
                        logger.warning(f"Certains thèmes originaux n'ont pas été assignés par le LLM: {missing_assignments}. Tentative d'assignation au premier méta-thème (0).")
                        fallback_meta_id = 0 # Assigner au premier méta-thème par défaut
                        if fallback_meta_id not in valid_meta_themes: # S'assurer que le méta-thème 0 existe
                             # Si même 0 n'existe pas (très improbable), prendre le premier dispo
                             fallback_meta_id = min(valid_meta_themes.keys()) if valid_meta_themes else 0
                             if fallback_meta_id not in valid_meta_themes and fallback_meta_id == 0: # Créer le 0 s'il manque et qu'on en a besoin
                                  valid_meta_themes[0] = "Méta-thème 1 (Généré)"


                        for missing_id_str in missing_assignments:
                             try:
                                valid_assignments[int(missing_id_str)] = fallback_meta_id
                             except ValueError: pass # Ignorer si l'ID n'est pas un int (ne devrait pas arriver)


                    # Si tout semble OK
                    if valid_meta_themes and valid_assignments:
                         logger.info(f"Méta-clustering réussi via LLM. {len(valid_meta_themes)} méta-thèmes définis.")
                         final_meta_theme_names = valid_meta_themes
                         final_theme_to_meta_assignment = valid_assignments
                         return final_meta_theme_names, final_theme_to_meta_assignment
                    else:
                         logger.error("Méta-clustering: JSON reçu mais invalide ou incomplet après validation.")

                else:
                    logger.error("Méta-clustering: La structure JSON ('meta_themes' ou 'assignments') est incorrecte ou manquante.")

            except json.JSONDecodeError as jde:
                logger.error(f"Méta-clustering: Erreur de décodage JSON lors du traitement de la réponse: {jde}")
        else:
            logger.error("Méta-clustering: Aucun bloc JSON trouvé dans la réponse du LLM.")

    except Exception as e:
        logger.error(f"Erreur grave lors du méta-clustering LLM: {e}", exc_info=True)

    # --- Fallback si le LLM échoue ---
    logger.warning("Méta-clustering LLM a échoué. Utilisation d'un regroupement simple comme fallback.")
    simple_meta_names = {}
    simple_assignments = {}

    # Créer un regroupement basique des thèmes
    original_theme_ids = sorted(clusters.keys())
    num_themes_to_group = len(original_theme_ids)
    # Répartir aussi équitablement que possible
    base_size = num_themes_to_group // target_count
    remainder = num_themes_to_group % target_count

    current_theme_index = 0
    for meta_id in range(target_count):
        size = base_size + (1 if meta_id < remainder else 0)
        simple_meta_names[meta_id] = f"Groupe Thématique {meta_id+1}"
        # Assigner les thèmes originaux à ce méta-thème
        for i in range(size):
            if current_theme_index < num_themes_to_group:
                original_id = original_theme_ids[current_theme_index]
                simple_assignments[original_id] = meta_id
                current_theme_index += 1
            else: break # Should not happen with correct calculation

    # S'assurer que tous les thèmes sont assignés (sécurité)
    assigned_count = len(simple_assignments)
    if assigned_count != num_themes_to_group:
         logger.error(f"Erreur dans le fallback de méta-clustering: {assigned_count} assignés vs {num_themes_to_group} attendus.")
         # Tenter de corriger en assignant les manquants au dernier groupe? Risqué.
         # Ou juste logguer l'erreur.

    logger.info(f"Fallback de méta-clustering terminé avec {len(simple_meta_names)} groupes.")
    return simple_meta_names, simple_assignments


# -----------------------------------
# Étape 8 : Compilation du tableau final (INCHANGÉ)
# -----------------------------------

def compile_results_with_meta(
    all_codes: List[Dict[str, Any]],
    code_to_cluster: Dict[str, int], # Mapping: code_text -> original_cluster_id
    theme_map: Dict[int, str],       # Mapping: original_cluster_id -> theme_name
    meta_theme_map: Dict[int, str],  # Mapping: meta_theme_id -> meta_theme_name
    theme_to_meta: Dict[int, int],   # Mapping: original_cluster_id -> meta_theme_id
) -> pd.DataFrame:
    """
    Génère un DataFrame pandas avec colonnes: Méta-thème, Thème, Code, Extrait
    """
    rows = []
    logger.info("Compilation des résultats finaux en DataFrame.")
    logger.debug(f"Mappings reçus: code->cluster: {len(code_to_cluster)} items, cluster->theme: {len(theme_map)} items, meta->name: {len(meta_theme_map)} items, theme->meta: {len(theme_to_meta)} items.")


    missing_assignments_count = 0
    codes_without_cluster = 0

    for code_dict in all_codes:
        code_text = code_dict.get("code")
        excerpt = code_dict.get("excerpt")

        if not code_text:
             logger.warning(f"Code manquant dans l'objet: {code_dict}. Ignoré.")
             continue

        # 1. Trouver l'ID du cluster original pour ce texte de code
        original_cluster_id = code_to_cluster.get(code_text, -1) # Utiliser -1 comme indicateur d'absence

        if original_cluster_id == -1:
            codes_without_cluster +=1
            theme_name = "Thème non assigné"
            meta_theme_name = "Méta-thème non assigné"
        else:
            # 2. Trouver le nom du thème original
            theme_name = theme_map.get(original_cluster_id, f"Thème Inconnu ({original_cluster_id})")

            # 3. Trouver l'ID du méta-thème pour ce thème original
            meta_theme_id = theme_to_meta.get(original_cluster_id, -1) # Utiliser -1 comme indicateur

            if meta_theme_id == -1:
                missing_assignments_count += 1
                meta_theme_name = "Méta-thème non assigné"
                 # Log seulement la première fois pour éviter le spam
                if missing_assignments_count == 1:
                    logger.warning(f"Le thème original ID {original_cluster_id} ('{theme_name}') n'a pas d'assignation à un méta-thème dans theme_to_meta map. Vérifiez l'étape de méta-clustering.")
            else:
                # 4. Trouver le nom du méta-thème
                meta_theme_name = meta_theme_map.get(meta_theme_id, f"Méta-thème Inconnu ({meta_theme_id})")

        rows.append(
            {"Méta-thème": meta_theme_name, "Thème": theme_name, "Code": code_text, "Extrait": excerpt}
        )

    if codes_without_cluster > 0:
         logger.warning(f"{codes_without_cluster} codes n'ont pas pu être associés à un cluster original (problème potentiel dans code_to_cluster map).")
    if missing_assignments_count > 0:
         logger.warning(f"Il y a eu {missing_assignments_count} instances où un thème original n'a pas pu être lié à un méta-thème (problème potentiel dans theme_to_meta map).")


    # Créer le DataFrame
    if not rows:
         logger.warning("Aucune donnée à compiler dans le DataFrame.")
         # Retourner un DataFrame vide avec les colonnes attendues
         return pd.DataFrame(columns=["Méta-thème", "Thème", "Code", "Extrait"])

    df = pd.DataFrame(rows)

    # Trier par méta-thème puis par thème pour une meilleure lisibilité
    # Gérer les cas où les noms sont des placeholders comme "Non assigné"
    # En les triant peut-être à la fin? Ou au début? Mettons-les au début.
    df['Méta-thème'] = pd.Categorical(df['Méta-thème'], ordered=True)
    df['Thème'] = pd.Categorical(df['Thème'], ordered=True)

    # Trier. pandas trie les catégories dans l'ordre où elles apparaissent par défaut,
    # sauf si on définit explicitement un ordre. Essayons le tri par défaut d'abord.
    try:
         df = df.sort_values(by=["Méta-thème", "Thème", "Code"])
    except Exception as sort_e:
         logger.warning(f"Échec du tri du DataFrame: {sort_e}. Le DataFrame non trié sera retourné.")


    logger.info(f"Compilation terminée. DataFrame créé avec {len(df)} lignes.")
    return df

def extract_structured_data(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convertit le DataFrame en liste d'objets structurés.
    Étape "Parser" de la méthodologie.
    """
    structured_data = []

    for _, row in df.iterrows():
        entry = {
            "meta": row.get("Méta-thème", "N/A"),
            "theme": row.get("Thème", "N/A"),
            "code": row.get("Code", "N/A"),
            "excerpt": row.get("Extrait", "N/A"),
        }
        structured_data.append(entry)

    logging.info(f"Données structurées extraites: {len(structured_data)} entrées")
    return structured_data


def plan_sections(structured_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Identifie les méta-thèmes uniques et génère un plan de sections initial.
    Étape "Planner" de la méthodologie.
    """
    sections = {}
    meta_themes = set(item["meta"] for item in structured_data)

    for meta in meta_themes:
        # Extraire les données pour ce méta-thème
        meta_data = [item for item in structured_data if item["meta"] == meta]
        themes = set(item["theme"] for item in meta_data)
        codes = set(item["code"] for item in meta_data)

        sections[meta] = {
            "title": meta,  # Le titre sera raffiné par l'IA
            "guiding_question": "",  # Sera généré par l'IA
            "themes": list(themes),
            "codes": list(codes),
            "excerpts": [item["excerpt"] for item in meta_data],
            "data": meta_data,
        }

    logging.info(f"Plan des sections créé: {len(sections)} méta-thèmes identifiés")
    return sections


# --- Extended Analysis Functions ---


def generate_planning_with_groq(
    client: Groq, sections: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Utilise Groq pour générer des titres et questions-guides pour chaque section.
    Inclut retry et gestion des erreurs API.
    Partie de l'étape "Planner".
    """
    updated_sections = sections.copy()

    system_prompt = """
    Tu es un sociologue expert en méthodologie qualitative.
    Tu dois générer un titre concis (2-5 mots) et une question-guide analytique pour une section d'analyse,
    en te basant sur les thèmes, codes et extraits fournis.
    La réponse doit être STRICTEMENT au format JSON.
    """

    for meta, section in updated_sections.items():
        themes_str = ", ".join(section["themes"])
        codes_str = ", ".join(section["codes"])
        excerpts_sample = section["excerpts"][
            :3
        ]  # Limite à quelques extraits pour éviter les prompts trop longs
        excerpts_str = "\n".join([f"- {e}" for e in excerpts_sample])

        # --- PROMPT ADAPTÉ POUR JSON ---
        user_prompt = f"""
Méta-thème: {meta}

Thèmes associés: {themes_str}

Codes associés: {codes_str}

Quelques extraits représentatifs:
{excerpts_str}

Pour cette section d'analyse, génère un titre concis et percutant ainsi qu'une question-guide analytique.
Réponds STRICTEMENT au format JSON comme ceci:
{{
  "title": "Titre concis",
  "guiding_question": "Question analytique?"
}}
"""
        # --- FIN PROMPT ADAPTÉ ---

        planning_data = None
        last_error = None

        for attempt in range(MAX_RETRIES):
            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    model=GROQ_MODEL,
                    temperature=0.7,
                    response_format={"type": "json_object"},
                )

                response_content = chat_completion.choices[0].message.content

                # --- VALIDATION JSON ---
                planning_data = json.loads(response_content)
                # Additional check: ensure expected keys are present
                if "title" in planning_data and "guiding_question" in planning_data:
                    logging.info(
                        f"Planification générée pour '{meta}' (Attempt {attempt + 1}): '{planning_data.get('title', 'N/A')}'"
                    )
                    break  # Exit retry loop on success
                else:
                    last_error = "Réponse JSON valide mais structure inattendue"
                    logging.warning(
                        f"Planification JSON inattendue pour '{meta}' (Attempt {attempt + 1}): {response_content}"
                    )

            except RateLimitError as e:
                delay = INITIAL_DELAY * (2**attempt)
                logging.warning(
                    f"Rate limit atteint lors de la planification pour '{meta}' (Attempt {attempt + 1}/{MAX_RETRIES}). Retrying in {delay:.2f}s. Error: {e}"
                )
                last_error = e
                time.sleep(delay)
            except APIStatusError as e:
                # Capture other API errors (like 400)
                logging.error(
                    f"Erreur API lors de la planification pour '{meta}' (Attempt {attempt + 1}/{MAX_RETRIES}). Code: {e.status_code}. Message: {e.response.text}"
                )
                last_error = e
                # For 400 errors due to prompt, retrying the same prompt won't help.
                # For others, maybe a single retry? Let's break for now as per plan's idea of fixing upstream.
                break  # Do not retry other API status errors automatically
            except json.JSONDecodeError as e:
                logging.error(
                    f"Erreur de décodage JSON lors de la planification pour '{meta}' (Attempt {attempt + 1}/{MAX_RETRIES}). Error: {e}. Raw response: {response_content}"
                )
                last_error = e
                break  # JSON decode error is usually a model issue, retrying might not help
            except Exception as e:
                logging.error(
                    f"Erreur inattendue lors de la planification pour '{meta}' (Attempt {attempt + 1}/{MAX_RETRIES}): {e}"
                )
                last_error = e
                break  # Catch other unexpected errors

        if (
            planning_data
            and "title" in planning_data
            and "guiding_question" in planning_data
        ):
            updated_sections[meta]["title"] = planning_data["title"]
            updated_sections[meta]["guiding_question"] = planning_data[
                "guiding_question"
            ]
        else:
            logging.error(
                f"Échec final de la planification pour '{meta}' après {MAX_RETRIES} tentatives. Utilisation des valeurs par défaut. Dernière erreur: {last_error}"
            )
            # Keep default values if the API call fails after retries

    return updated_sections


def generate_section_content_with_groq(
    client: Groq, section: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Utilise Groq pour générer le contenu d'une section avec CoT et paragraphe initial.
    Inclut retry et gestion des erreurs API.
    Étape "Thinker" de la méthodologie.
    """
    meta = section.get(
        "title", section.get("meta", "Section inconnue")
    )  # Use refined title if available
    question = section.get("guiding_question", "")
    themes_str = ", ".join(section["themes"])
    codes_str = ", ".join(section["codes"])

    # Sélectionner tous les extraits pour cette section
    excerpts_str = "\n".join([f"- {e}" for e in section["excerpts"]])

    system_prompt = """
    Tu es un sociologue expert en analyse qualitative. Ta tâche est de générer:
    1. Une brève chaîne de pensée (CoT) qui relie logiquement les codes et extraits fournis
    2. Un paragraphe fluide (100-150 mots) qui synthétise ces données et intègre au moins un extrait complet

    Ton analyse doit être nuancée, rigoureuse et basée sur les données fournies.
    La réponse doit être STRICTEMENT au format JSON.
    """

    # --- PROMPT ADAPTÉ POUR JSON ---
    user_prompt = f"""
Section: {meta}
Question analytique: {question}

Thèmes: {themes_str}

Codes: {codes_str}

Extraits disponibles:
{excerpts_str}

Génère une chaîne de pensée et un paragraphe synthétique.
Réponds STRICTEMENT au format JSON comme ceci:
{{
  "chain_of_thought": "Ta réflexion analytique (40-60 mots)",
  "paragraph": "Ton paragraphe synthétique (100-150 mots incluant au moins un extrait complet entre guillemets)"
}}
"""
    # --- FIN PROMPT ADAPTÉ ---

    content_data = None
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=GROQ_MODEL,
                temperature=TEMPERATURE,
                response_format={"type": "json_object"},
            )

            response_content = chat_completion.choices[0].message.content

            # --- VALIDATION JSON ---
            content_data = json.loads(response_content)
            # Additional check: ensure expected keys are present
            if "chain_of_thought" in content_data and "paragraph" in content_data:
                logging.info(
                    f"Contenu initial généré pour la section '{meta}' (Attempt {attempt + 1})"
                )
                break  # Exit retry loop on success
            else:
                last_error = "Réponse JSON valide mais structure inattendue"
                logging.warning(
                    f"Contenu initial JSON inattendu pour '{meta}' (Attempt {attempt + 1}): {response_content}"
                )

        except RateLimitError as e:
            delay = INITIAL_DELAY * (2**attempt)
            logging.warning(
                f"Rate limit atteint lors de la génération de contenu pour '{meta}' (Attempt {attempt + 1}/{MAX_RETRIES}). Retrying in {delay:.2f}s. Error: {e}"
            )
            last_error = e
            time.sleep(delay)
        except APIStatusError as e:
            logging.error(
                f"Erreur API lors de la génération de contenu pour '{meta}' (Attempt {attempt + 1}/{MAX_RETRIES}). Code: {e.status_code}. Message: {e.response.text}"
            )
            last_error = e
            break  # Do not retry other API status errors automatically
        except json.JSONDecodeError as e:
            logging.error(
                f"Erreur de décodage JSON lors de la génération de contenu pour '{meta}' (Attempt {attempt + 1}/{MAX_RETRIES}). Error: {e}. Raw response: {response_content}"
            )
            last_error = e
            break  # JSON decode error is usually a model issue, retrying might not help
        except Exception as e:
            logging.error(
                f"Erreur inattendue lors de la génération de contenu pour '{meta}' (Attempt {attempt + 1}/{MAX_RETRIES}): {e}"
            )
            last_error = e
            break  # Catch other unexpected errors

    if (
        content_data
        and "chain_of_thought" in content_data
        and "paragraph" in content_data
    ):
        section["chain_of_thought"] = content_data["chain_of_thought"]
        section["initial_paragraph"] = content_data["paragraph"]
    else:
        logging.error(
            f"Échec final de la génération de contenu pour '{meta}' après {MAX_RETRIES} tentatives. Utilisation des valeurs par défaut. Dernière erreur: {last_error}"
        )
        section["chain_of_thought"] = "Analyse non disponible en raison d'une erreur."
        section["initial_paragraph"] = "Contenu non disponible en raison d'une erreur."

    return section


def review_and_improve_with_groq(
    client: Groq, section: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Utilise Groq pour critiquer et améliorer le paragraphe initial avec retry et gestion des erreurs API.
    Étape "Critic" de la méthodologie.
    """
    meta = section.get(
        "title", section.get("meta", "Section inconnue")
    )  # Use refined title if available
    paragraph = section.get("initial_paragraph", "")

    # Vérifier si le paragraphe initial existe et est significatif, sinon créer un fallback pour la révision
    if not paragraph or len(paragraph) < 30:
        logging.warning(
            f"Paragraphe initial insuffisant pour '{meta}', création d'un paragraphe de secours pour la révision."
        )
        if section["excerpts"]:
            sample_excerpt = section["excerpts"][0]
            paragraph_for_review = f"L'analyse des témoignages sur {meta} révèle des perspectives intéressantes. Comme l'illustre cet extrait: \"{sample_excerpt}\", les étudiants développent une approche pragmatique de l'IA dans ce contexte."
        else:
            paragraph_for_review = f"L'analyse relative à {meta} met en évidence des pratiques spécifiques des étudiants concernant l'usage de l'IA dans leur parcours académique, entre gain d'efficacité et questionnements éthiques."
    else:
        paragraph_for_review = paragraph  # Use the generated paragraph

    # Sélectionner soigneusement les extraits pertinents pour le prompt de révision
    relevant_excerpts = []
    for excerpt in section["excerpts"]:
        if len(excerpt) > 15:  # Extraits significatifs
            relevant_excerpts.append(excerpt)
        if (
            len(relevant_excerpts) >= 3
        ):  # Limiter à 3 extraits pertinents dans le prompt
            break

    # S'il n'y a pas assez d'extraits pertinents (>15 chars), prendre les premiers disponibles (jusqu'à 3)
    if len(relevant_excerpts) < 2 and len(section["excerpts"]) > 0:
        # Avoid including the same short excerpts multiple times if they are already in relevant_excerpts
        for excerpt in section["excerpts"][:3]:
            if excerpt not in relevant_excerpts:
                relevant_excerpts.append(excerpt)

    excerpts_str = "\n".join([f"- {e}" for e in relevant_excerpts])

    system_prompt = """
    Tu es un réviseur critique expert en méthodologie qualitative. Ta tâche est d'examiner un paragraphe d'analyse,
    d'identifier 1-2 points à améliorer (fidélité aux données, clarté, nuance) et de proposer une version améliorée.

    Ton paragraphe amélioré DOIT:
    1. Faire 100-150 mots
    2. Intégrer AU MOINS un extrait complet entre guillemets doubles ("...") (préférablement parmi les extraits fournis ou ceux que tu juges pertinents)
    3. Maintenir la cohérence thématique avec le titre de la section
    4. Être fidèle aux données et éviter toute surinterprétation
    5. Répondre STRICTEMENT au format JSON.
    """

    # --- PROMPT ADAPTÉ POUR JSON ---
    user_prompt = f"""
Titre de section: {meta}

Paragraphe à évaluer:
"{paragraph_for_review}"

Extraits originaux (pour référence):
{excerpts_str}

1. Identifie 1 ou 2 points d'amélioration précis (fidélité, clarté, nuance)
2. Propose une version améliorée du paragraphe (100-150 mots) qui intègre AU MOINS un extrait complet entre guillemets doubles.

Réponds STRICTEMENT au format JSON comme ceci:
{{
  "critique": ["Point d'amélioration 1", "Point d'amélioration 2 (optionnel)"],
  "improved_paragraph": "Paragraphe amélioré (100-150 mots avec au moins un extrait entre guillemets)"
}}
"""
    # --- FIN PROMPT ADAPTÉ ---

    review_data = None
    last_error = None

    # MAX_RETRIES est déjà défini globalement
    for attempt in range(MAX_RETRIES):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=GROQ_MODEL,
                temperature=0.6,  # Légèrement inférieur pour une critique plus factuelle
                response_format={"type": "json_object"},
            )

            response_content = chat_completion.choices[0].message.content

            # --- VALIDATION JSON ---
            review_data = json.loads(response_content)
            improved_paragraph = review_data.get("improved_paragraph", "")

            # Vérification supplémentaire de qualité: s'assurer qu'il y a au moins un extrait entre guillemets doubles et que le paragraphe est significatif
            if improved_paragraph and len(improved_paragraph) > 50:
                # Check for double quotes explicitly
                if '"' in improved_paragraph:
                    logging.info(
                        f"Révision effectuée avec succès pour la section '{meta}' (Attempt {attempt + 1})"
                    )
                    break  # Exit retry loop on success
                else:
                    last_error = "Paragraphe amélioré valide mais pas d'extrait entre guillemets doubles"
                    logging.warning(
                        f"Révision OK mais sans extrait double-guillemet pour '{meta}' (Attempt {attempt + 1})"
                    )
            else:
                last_error = "Paragraphe amélioré insuffisant ou manquant"
                logging.warning(
                    f"Paragraphe amélioré insuffisant/manquant pour '{meta}' (Attempt {attempt + 1})"
                )

        except RateLimitError as e:
            delay = INITIAL_DELAY * (2**attempt)
            logging.warning(
                f"Rate limit atteint lors de la révision pour '{meta}' (Attempt {attempt + 1}/{MAX_RETRIES}). Retrying in {delay:.2f}s. Error: {e}"
            )
            last_error = e
            time.sleep(delay)
        except APIStatusError as e:
            logging.error(
                f"Erreur API lors de la révision pour '{meta}' (Attempt {attempt + 1}/{MAX_RETRIES}). Code: {e.status_code}. Message: {e.response.text}"
            )
            last_error = e
            break  # Do not retry other API status errors automatically
        except json.JSONDecodeError as e:
            logging.error(
                f"Erreur de décodage JSON lors de la révision pour '{meta}' (Attempt {attempt + 1}/{MAX_RETRIES}). Error: {e}. Raw response: {response_content}"
            )
            last_error = e
            break  # JSON decode error is usually a model issue, retrying might not help
        except Exception as e:
            logging.error(
                f"Erreur inattendue lors de la révision pour '{meta}' (Attempt {attempt + 1}/{MAX_RETRIES}): {e}"
            )
            last_error = e
            break  # Catch other unexpected errors

    if (
        review_data
        and "improved_paragraph" in review_data
        and len(review_data["improved_paragraph"]) > 50
    ):
        # Final check for quote even if model didn't include it in first try, add one if possible
        final_paragraph = review_data["improved_paragraph"]
        if '"' not in final_paragraph and relevant_excerpts:
            best_excerpt = max(
                relevant_excerpts, key=len
            )  # Choose the longest available excerpt
            if len(best_excerpt) > 15:  # Only add if excerpt is significant
                final_paragraph = (
                    final_paragraph.strip()
                    + f' Comme l\'illustre cet extrait : "{best_excerpt}".'
                )
                logging.warning(
                    f"Ajout manuel d'un extrait entre guillemets pour '{meta}'."
                )

        section["critique"] = review_data.get("critique", ["Critique non disponible."])
        section["final_paragraph"] = (
            final_paragraph.strip()
        )  # Ensure no leading/trailing whitespace

    else:
        logging.error(
            f"Échec final de la révision pour '{meta}' après {MAX_RETRIES} tentatives. Génération d'un paragraphe de secours. Dernière erreur: {last_error}"
        )
        # If all attempts failed, create a fallback improved paragraph manually
        try:
            # Attempt to include a relevant excerpt in the fallback
            if relevant_excerpts:
                best_excerpt = max(
                    relevant_excerpts, key=len
                )  # Choose the longest available excerpt
                if len(best_excerpt) > 15:  # Only add if excerpt is significant
                    section["final_paragraph"] = (
                        f"L'analyse des données concernant {meta} révèle des perspectives importantes sur l'usage de l'IA par les étudiants. Comme l'illustre cet extrait significatif : \"{best_excerpt}\", les participants développent des stratégies adaptatives face à ces technologies. Cette dimension s'inscrit dans une réflexion plus large sur l'évolution des pratiques académiques et l'autonomie intellectuelle à l'ère numérique."
                    )
                else:
                    # Fallback to initial paragraph if no good excerpts
                    section["final_paragraph"] = paragraph_for_review.strip()
            else:
                # Fallback to initial paragraph if no excerpts
                section["final_paragraph"] = paragraph_for_review.strip()

            section["critique"] = [
                "Amélioration automatique suite à une erreur technique."
            ]
            logging.info(f"Paragraphe de secours généré pour la section '{meta}'")
        except Exception:
            # Ultimate fallback: use the initial paragraph without modification
            section["final_paragraph"] = paragraph_for_review.strip()
            section["critique"] = [
                "Critique et amélioration non disponibles en raison d'erreurs successives."
            ]
            logging.warning(
                f"Utilisation du paragraphe initial ({len(paragraph_for_review.strip())} chars) pour '{meta}' suite à des erreurs critiques"
            )

    return section


def generate_introduction_conclusion_with_groq(
    client: Groq, sections: Dict[str, Dict[str, Any]]
) -> Tuple[str, str]:
    """
    Génère l'introduction et la conclusion de la synthèse avec retry et gestion des erreurs API.
    Partie de l'étape "Synthesizer".
    """
    # Extraire les titres des sections (raffinés par l'IA si la planification a réussi)
    section_titles = [section.get("title", meta) for meta, section in sections.items()]

    # Extraire des extraits significatifs pour chaque section (pour enrichir l'intro/conclusion)
    all_excerpts = []
    for section in sections.values():
        section_excerpts = []
        # S'assurer que les extraits ne sont pas trop longs pour le prompt
        for excerpt in section["excerpts"]:
            if len(excerpt) > 40:  # Limiter les extraits trop longs pour le prompt
                section_excerpts.append(
                    excerpt[:200].rsplit(" ", 1)[0] + "..."
                )  # Truncate nicely
            elif len(excerpt) > 15:
                section_excerpts.append(excerpt)

            if (
                len(section_excerpts) >= 2
            ):  # Maximum 2 extraits pertinents par section dans le prompt
                break
        all_excerpts.extend(section_excerpts)

    sample_excerpts = all_excerpts[
        :10
    ]  # Limiter le nombre total d'extraits dans le prompt
    excerpts_str = "\n".join([f"- {e}" for e in sample_excerpts])
    sections_str = ", ".join(section_titles)

    # Créer une introduction par défaut en cas d'échec de l'API
    default_intro = f"""
L'avènement de l'intelligence artificielle, notamment à travers des outils comme ChatGPT, transforme profondément les pratiques des étudiants dans l'enseignement supérieur. Cette synthèse explore les usages, perceptions et implications de ces technologies dans le contexte académique. À travers l'analyse d'entretiens qualitatifs, nous avons identifié plusieurs thématiques centrales, structurées autour des sections suivantes : {sections_str}. Ces dimensions permettent de comprendre comment les étudiants intègrent l'IA dans leurs pratiques quotidiennes, entre gain d'efficacité et questionnements éthiques, et les défis que cela pose pour l'avenir de l'enseignement supérieur.
"""

    # Créer une conclusion par défaut en cas d'échec de l'API
    default_conclusion = f"""
Face à l'intégration croissante de l'IA dans les pratiques académiques, les perspectives se concentrent sur l'adaptation nécessaire de l'enseignement supérieur. Cela inclut la refonte des méthodes d'évaluation pour mieux valoriser la pensée critique et l'analyse plutôt que la simple restitution, la formation des étudiants à une utilisation éthique et discernée des outils d'IA, et une réflexion continue sur la manière de favoriser une collaboration fructueuse entre l'intelligence humaine et artificielle dans le processus d'apprentissage et de recherche. Ces ajustements sont essentiels pour préparer les étudiants aux défis et opportunités de l'ère numérique.
"""

    system_prompt = """
    Tu es un sociologue expert en analyse qualitative. Ta tâche est de générer:
    1. Une introduction captivante pour une synthèse narrative (100-150 mots)
    2. Une conclusion "Perspectives" concise et tournée vers l'avenir (environ 50-80 mots, 3-4 phrases)

    Ces textes doivent encadrer une analyse des usages de l'IA (notamment ChatGPT) par des étudiants dans leur parcours académique.

    L'introduction doit:
    - Présenter le contexte de l'étude et l'importance du sujet (IA dans l'enseignement supérieur)
    - Mentionner brièvement la méthodologie qualitative utilisée (analyse d'entretiens)
    - Présenter les principaux thèmes ou sections qui seront abordés (utiliser la liste fournie)
    - Indiquer l'objectif général de la synthèse.

    La conclusion "Perspectives" doit:
    - Proposer 3-4 pistes de réflexion, implications ou recommandations concrètes basées sur une analyse des usages (formation, évaluation, complémentarité homme-machine, etc.)
    - Être orientée vers des recommandations pratiques pour l'avenir de l'enseignement supérieur face à l'IA.
    - Ne pas simplement résumer l'analyse déjà présentée dans les sections.
    - Répondre STRICTEMENT au format JSON.
    """

    # --- PROMPT ADAPTÉ POUR JSON ---
    user_prompt = f"""
La synthèse sociologique explore l'usage de l'IA par les étudiants et comprend les sections d'analyse suivantes: {sections_str}

Voici des extraits représentatifs d'entretiens avec des étudiants pour contexte:
{excerpts_str}

Génère:
1. Une introduction complète (100-150 mots) pour cette synthèse.
2. Une conclusion "Perspectives" concise (environ 50-80 mots) qui propose des pistes pour l'avenir de l'enseignement supérieur face à l'IA.

Réponds STRICTEMENT au format JSON comme ceci:
{{
  "introduction": "Texte d'introduction (100-150 mots)",
  "conclusion": "Texte de conclusion (environ 50-80 mots)"
}}
"""
    # --- FIN PROMPT ADAPTÉ ---

    bookends_data = None
    last_error = None

    # MAX_RETRIES est déjà défini globalement
    for attempt in range(MAX_RETRIES):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=GROQ_MODEL,
                temperature=0.6,  # Température réduite pour plus de cohérence
                response_format={"type": "json_object"},
            )

            response_content = chat_completion.choices[0].message.content

            # --- VALIDATION JSON ---
            bookends_data = json.loads(response_content)

            # Additional check: ensure expected keys are present and content is significant
            if (
                bookends_data
                and "introduction" in bookends_data
                and "conclusion" in bookends_data
                and len(bookends_data.get("introduction", "")) > 50
                and len(bookends_data.get("conclusion", "")) > 30
            ):
                logging.info(
                    f"Introduction et conclusion générées avec succès (Attempt {attempt + 1})"
                )
                return (
                    bookends_data["introduction"].strip(),
                    bookends_data["conclusion"].strip(),
                )
            else:
                last_error = "Réponse JSON valide mais structure ou contenu insuffisant"
                logging.warning(
                    f"Intro/Conclusion JSON inattendue ou insuffisante (Attempt {attempt + 1}): {response_content}"
                )

        except RateLimitError as e:
            delay = INITIAL_DELAY * (2**attempt)
            logging.warning(
                f"Rate limit atteint lors de la génération Intro/Conclusion (Attempt {attempt + 1}/{MAX_RETRIES}). Retrying in {delay:.2f}s. Error: {e}"
            )
            last_error = e
            time.sleep(delay)
        except APIStatusError as e:
            logging.error(
                f"Erreur API lors de la génération Intro/Conclusion (Attempt {attempt + 1}/{MAX_RETRIES}). Code: {e.status_code}. Message: {e.response.text}"
            )
            last_error = e
            break  # Do not retry other API status errors automatically
        except json.JSONDecodeError as e:
            logging.error(
                f"Erreur de décodage JSON lors de la génération Intro/Conclusion (Attempt {attempt + 1}/{MAX_RETRIES}). Error: {e}. Raw response: {response_content}"
            )
            last_error = e
            # Keep the manual extraction attempt from original code as a fallback here
            try:
                # Attempt manual extraction if JSON parsing fails
                if (
                    response_content
                    and "introduction" in response_content.lower()
                    and "conclusion" in response_content.lower()
                ):
                    parts = response_content.split("conclusion", 1)
                    intro_part = (
                        parts[0].lower().split("introduction", 1)[1]
                    )  # Split again on introduction after splitting on conclusion
                    conclu_part = parts[1]

                    # Basic cleaning
                    intro_clean = intro_part.strip().strip('":{} \n').strip()
                    conclu_clean = conclu_part.strip().strip('":{} \n').strip()

                    if len(intro_clean) > 50 and len(conclu_clean) > 30:
                        logging.warning(
                            f"Récupération manuelle partielle Intro/Conclusion après échec JSON pour '{meta}'."
                        )
                        return intro_clean, conclu_clean
                else:
                    logging.warning(
                        f"Échec de la récupération manuelle après erreur JSON pour '{meta}'."
                    )
            except Exception as manual_e:
                logging.warning(
                    f"Erreur lors de la tentative de récupération manuelle: {manual_e}"
                )

            break  # Break retry loop after JSON error and manual recovery attempt

        except Exception as e:
            logging.error(
                f"Erreur inattendue lors de la génération de l'introduction et de la conclusion (Attempt {attempt + 1}/{MAX_RETRIES}): {e}"
            )
            last_error = e
            break  # Catch other unexpected errors

    # If all attempts failed or critical errors occurred
    logging.error(
        f"Impossible de générer l'introduction et la conclusion après {MAX_RETRIES} tentatives. Utilisation des textes par défaut. Dernière erreur: {last_error}"
    )
    return default_intro.strip(), default_conclusion.strip()


def assemble_final_report(
    introduction: str, section_contents: Dict[str, Dict[str, Any]], conclusion: str
) -> str:
    """
    Assemble les différentes parties en un rapport cohérent.
    Étape "Synthesizer" finale de la méthodologie.
    """
    report = "# SYNTHÈSE SOCIOLOGIQUE : L'USAGE DE L'IA CHEZ LES ÉTUDIANTS\n\n"

    # Add introduction
    report += "## Introduction\n\n"
    report += introduction + "\n\n"

    # Add each section with its title and content
    for meta_theme_key, section in section_contents.items():
        # Use the potentially refined title, fallback to original meta key
        title_to_use = section.get("title", meta_theme_key)
        report += f"## {title_to_use}\n\n"
        # Use the final_paragraph from revision, fallback to initial if revision failed, ultimate fallback to a default
        final_paragraph = section.get(
            "final_paragraph",
            section.get(
                "initial_paragraph", "Contenu non disponible en raison d'une erreur."
            ),
        )
        report += final_paragraph + "\n\n"

    # Add conclusion
    report += "## Perspectives\n\n"
    report += conclusion + "\n\n"

    return report



from typing import Dict
from src.state import GraphState

import pandas as pd


def extract_node(state: GraphState) -> Dict:
    raw_text = extract_text_from_pdf(state["pdf_path"])
    return {"raw_text": raw_text}


def clean_node(state: GraphState) -> Dict:
    clean = clean_text(state["raw_text"])
    return {"clean_text": clean}


def segment_node(state: GraphState) -> Dict:
    segments = segment_text(state["clean_text"])
    return {"segments": segments}


def analyze_node(state: GraphState) -> Dict:
    api_key = load_api_key()
    client = create_groq_client(api_key)
    model = state.get("model_name", "llama3-70b-8192")
    all_codes = []
    for seg in state["segments"]:
        codes = analyze_and_code_segment(client, model, seg)
        if codes:
            for c in codes:
                c["segment"] = seg  # Pour traçabilité
            all_codes.extend(codes)
    return {"all_codes": all_codes}


def judge_node(state: GraphState) -> Dict:
    api_key = load_api_key()
    client = create_groq_client(api_key)
    model = "llama3-70b-8192"
    validated = []
    for seg in state["segments"]:
        codes_for_seg = [c for c in state["all_codes"] if c.get("segment") == seg]
        validated_codes = judge_codes(client, model, seg, codes_for_seg)
        if validated_codes:
            validated.extend(validated_codes)
    return {"validated_segments": validated}


def cluster_node(state: GraphState) -> Dict:
    clusters, code_to_cluster = cluster_codes_limited(
        state["validated_segments"], state["max_themes"]
    )
    return {"clusters": clusters, "code_to_cluster": code_to_cluster}


def label_node(state: GraphState) -> Dict:
    api_key = load_api_key()
    client = create_groq_client(api_key)
    model = state.get("model_name", "llama3-70b-8192")
    theme_labels = label_themes(client, model, state["clusters"])
    return {"theme_labels": theme_labels}


def meta_cluster_node(state: GraphState) -> Dict:
    api_key = load_api_key()
    client = create_groq_client(api_key)
    model = state.get("model_name", "llama3-70b-8192")
    meta_theme_map, theme_to_meta = meta_cluster_themes(
        client,
        model,
        state["clusters"],
        state["theme_labels"],
        state["max_themes"]
    )
    return {
        "meta_theme_map": meta_theme_map,
        "theme_to_meta": theme_to_meta
    }


def compile_node(state: GraphState) -> Dict:
    # Fallback si le méta-clustering n'a pas été fait
    meta_theme_map = state.get("meta_theme_map")
    theme_to_meta = state.get("theme_to_meta")

    if meta_theme_map is None or theme_to_meta is None:
        # Pas de méta-clustering : on considère chaque thème comme son propre méta-thème
        meta_theme_map = {k: v for k, v in state["theme_labels"].items()}
        theme_to_meta = {k: k for k in state["theme_labels"].keys()}

    df = compile_results_with_meta(
        state["validated_segments"],
        state["code_to_cluster"],
        state["theme_labels"],
        meta_theme_map,
        theme_to_meta
    )

    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "final_results.csv")
    df.to_csv(output_file, index=False, encoding='utf-8-sig')

    return {
        "final_report": f"Résultats compilés dans {output_file}",
        "output_files": {"csv": output_file}
    }

__all__ = [
    'extract_node',
    'clean_node',
    'segment_node',
    'analyze_node',
    'judge_node',
    'cluster_node',
    'label_node',
    'meta_cluster_node',
    'compile_node'
]
def generate_full_report(client: Groq, csv_path: str, output_txt_path: str):
    """
    Génère un rapport sociologique complet depuis un CSV de clustering.
    """
    df = load_data(csv_path)
    structured_data = extract_structured_data(df)
    
    # Planification
    sections = plan_sections(structured_data)
    sections_with_planning = generate_planning_with_groq(client, sections)
    
    # Contenu + critique
    for meta_key, section in sections_with_planning.items():
        section = generate_section_content_with_groq(client, section)
        section = review_and_improve_with_groq(client, section)
        sections_with_planning[meta_key] = section
    
    # Introduction / Conclusion
    intro, conclu = generate_introduction_conclusion_with_groq(client, sections_with_planning)
    
    # Assemblage final
    report = assemble_final_report(intro, sections_with_planning, conclu)

    # Sauvegarde
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"Rapport complet sauvegardé dans {output_txt_path}")
import pandas as pd
def load_data(filepath: str) -> pd.DataFrame:
    """Charge un CSV en DataFrame avec validation basique."""
    df = pd.read_csv(filepath)
    if df.empty:
        raise ValueError(f"Le fichier {filepath} est vide.")
    return df