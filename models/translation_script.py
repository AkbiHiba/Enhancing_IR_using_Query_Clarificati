#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour traduire un dataset JSON (dev_with_evidence_articles.json) de l'anglais vers le français
en utilisant des modèles de traduction avancés.

Ce script crée un fichier de sortie au fur et à mesure que les traductions sont effectuées
pour permettre de visualiser la progression de la traduction.
"""

import json
import os
import time
import torch
from tqdm import tqdm
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import gc
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Télécharger 'punkt' si ce n’est pas déjà fait
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class Config:
    # Chemins des fichiers
    input_path = "qulac2_329_358_top4to20_en.json"
    output_path = "qulac2_329_358_top4to20_fr.json"
    temp_path = "qulac2_329_358_top4to20_fr_temp.json"
    
    # Paramètres de traduction et de segmentation
    model_name = "facebook/nllb-200-distilled-600M"  # Updated model
    cache_dir = "tmp/huggingfacecache"
    
    # Paramètres pour gérer les textes longs
    max_tokens = 450  # Limite de tokens pour le modèle (marge de sécurité)
    max_segment_chars = 200  # Taille optimale pour un segment
    min_segment_chars = 50  # Taille minimale d'un segment
    overlap_tokens = 20  # Nombre de tokens de chevauchement
    
    # Liste des champs à exclure de la traduction
    NON_TRANSLATABLE_FIELDS = ["id","articles_html_text"]  
    
    # Paramètres système
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Paramètres de sauvegarde et progression
    save_frequency = 1  # Sauvegarder après chaque article traduit
    show_progress = True  # Afficher la progression détaillée

class TextSegmenter:
    """Classe pour gérer la segmentation intelligente des textes longs"""
    
    @staticmethod
    def split_into_sentences(text):
        """Divise le texte en phrases"""
        try:
            return sent_tokenize(text)
        except Exception as e:
            print(f"Erreur lors de la division en phrases: {e}")
            return [text]
    
    @staticmethod
    def merge_short_sentences(sentences, min_length=Config.min_segment_chars):
        """Fusionne les phrases courtes"""
        merged = []
        current = ""
        
        for sent in sentences:
            if len(current) + len(sent) < min_length:
                current = (current + " " + sent).strip()
            else:
                if current:
                    merged.append(current)
                current = sent
        
        if current:
            merged.append(current)
        
        return merged
    
    @staticmethod
    def split_long_sentences(sentence, max_length=Config.max_segment_chars):
        """Divise les phrases trop longues en segments plus petits"""
        if len(sentence) <= max_length:
            return [sentence]
        
        words = word_tokenize(sentence)
        segments = []
        current_segment = []
        current_length = 0
        
        for word in words:
            word_len = len(word) + 1  # +1 pour l'espace
            if current_length + word_len > max_length and current_segment:
                segments.append(" ".join(current_segment))
                current_segment = [word]
                current_length = word_len
            else:
                current_segment.append(word)
                current_length += word_len
        
        if current_segment:
            segments.append(" ".join(current_segment))
        
        return segments
    
    @staticmethod
    def create_overlapping_segments(segments, overlap_size=Config.overlap_tokens):
        """Crée des segments avec chevauchement pour améliorer la cohérence"""
        if not segments:
            return []
        
        overlapped = []
        previous_end = []
        
        for segment in segments:
            words = word_tokenize(segment)
            
            # Ajouter les mots de chevauchement au début
            if previous_end:
                words = previous_end + words
            
            # Garder les derniers mots pour le prochain segment
            previous_end = words[-overlap_size:] if len(words) > overlap_size else words
            
            overlapped.append(" ".join(words))
        
        return overlapped
    
    @staticmethod
    def optimize_segments(segments):
        """Optimise les segments pour la traduction"""
        optimized = []
        current = ""
        
        for segment in segments:
            if not segment.strip():
                continue
                
            if len(current) + len(segment) < Config.max_segment_chars:
                current = (current + " " + segment).strip()
            else:
                if current:
                    optimized.append(current)
                current = segment
        
        if current:
            optimized.append(current)
        
        return optimized

class TextTranslator:
    """Classe pour gérer la traduction des textes"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.translator = None
        self.segmenter = TextSegmenter()
        self.src_lang = "eng_Latn"  # Anglais
        self.tgt_lang = "fra_Latn"  # Français
        self.load_model()
    
    def load_model(self):
        """Charge le modèle de traduction"""
        print(f"Chargement du modèle {Config.model_name}...")
        
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                Config.model_name,
                cache_dir=Config.cache_dir
            ).to(Config.device)
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                Config.model_name,
                cache_dir=Config.cache_dir
            )
            
            # Configuration correcte pour NLLB
            self.translator = pipeline(
                "translation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if Config.use_cuda else -1
            )
            
            print(f"Modèle chargé avec succès sur {Config.device}")
            
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {e}")
            raise
    
    def translate_segment(self, text):
        """Traduit un segment de texte"""
        if not text or not text.strip():
            return ""
        
        try:
            # Passage explicite des langues source et cible à chaque traduction
            result = self.translator(
                text, 
                src_lang=self.src_lang,
                tgt_lang=self.tgt_lang,
                max_length=Config.max_tokens
            )
            return result[0]["translation_text"]
        except Exception as e:
            print(f"Erreur lors de la traduction du segment: {e}")
            print(f"Texte: {text[:50]}...")  # Affiche le début du texte problématique
            return text
    
    def translate_long_text(self, text, display=True):
        """Traduit un texte long en le divisant en segments"""
        if not text or not text.strip():
            return ""
        
        try:
            # 1. Diviser en phrases
            sentences = self.segmenter.split_into_sentences(text)
            
            # 2. Fusionner les phrases courtes
            merged = self.segmenter.merge_short_sentences(sentences)
            
            # 3. Diviser les phrases trop longues
            segments = []
            for sentence in merged:
                if len(sentence) > Config.max_segment_chars:
                    segments.extend(self.segmenter.split_long_sentences(sentence))
                else:
                    segments.append(sentence)
            
            # 4. Créer des segments avec chevauchement
            overlapped = self.segmenter.create_overlapping_segments(segments)
            
            # 5. Optimiser les segments
            optimized = self.segmenter.optimize_segments(overlapped)
            
            # 6. Traduire chaque segment avec barre de progression
            translated_segments = []
            iterator = tqdm(optimized, desc="Traduction", disable=not display)
            for i, segment in enumerate(iterator):
                translated = self.translate_segment(segment)
                translated_segments.append(translated)
                
                # Nettoyage périodique de la mémoire
                if (i + 1) % 10 == 0:
                    self.clean_gpu_memory()
            
            # 7. Post-traitement pour améliorer la cohérence
            final_translation = self.post_process_translation(translated_segments)
            
            return final_translation
            
        except Exception as e:
            print(f"Erreur lors de la traduction du texte long: {e}")
            return text
    
    def post_process_translation(self, segments):
        """Post-traitement pour supprimer les parties dupliquées entre segments"""
        if not segments:
            return ""
        
        processed_segments = [segments[0]]
        for segment in segments[1:]:
            prev_words = word_tokenize(processed_segments[-1])
            curr_words = word_tokenize(segment)
            
            # Détecter le chevauchement jusqu'à 20 tokens
            max_overlap = min(20, len(prev_words), len(curr_words))
            overlap_size = 0
            for i in range(max_overlap, 0, -1):
                if prev_words[-i:] == curr_words[:i]:
                    overlap_size = i
                    break
            
            # Supprimer le chevauchement du début du segment courant
            cleaned_segment = " ".join(curr_words[overlap_size:])
            processed_segments.append(cleaned_segment)
        
        return " ".join(processed_segments)
    
    def clean_gpu_memory(self):
        """Nettoie la mémoire GPU"""
        if Config.use_cuda:
            torch.cuda.empty_cache()
            gc.collect()

class JsonTranslator:
    """Classe principale pour la traduction du fichier JSON"""
    
    def __init__(self):
        self.translator = TextTranslator()
        self.current_data = None  # Pour stocker les données en cours
    
    def translate_field(self, value, field_name=None, path=None):
        """Traduit un champ du JSON et met à jour le fichier. Seuls les champs listés dans Config.TRANSLATABLE_FIELDS sont traduits et conservés."""
        if path is None:
            path = []
        
        # Utiliser la logique par champs exclus
        if field_name is not None and field_name in Config.NON_TRANSLATABLE_FIELDS:
            return value
            
        if isinstance(value, str):
            translated = self.translator.translate_long_text(value)
            # Mettre à jour les données actuelles et sauvegarder
            self.update_and_save(path, translated)
            return translated
        elif isinstance(value, list):
            translated_list = []
            for i, item in enumerate(value):
                t = self.translate_field(item, field_name=None, path=path + [i])
                if t is not None:
                    translated_list.append(t)
            return translated_list
        elif isinstance(value, dict):
            translated_dict = {}
            for k, v in value.items():
                t = self.translate_field(v, field_name=k, path=path + [k])
                if t is not None:
                    translated_dict[k] = t
            return translated_dict
        return value
    
    def update_and_save(self, path, value):
        """Met à jour les données actuelles et sauvegarde"""
        if not path:
            return
        try:
            current = self.current_data
            for i, key in enumerate(path[:-1]):
                if isinstance(current, list):
                    while len(current) <= key:
                        next_key = path[i+1] if i+1 < len(path) else None
                        container = [] if isinstance(next_key, int) else {}
                        current.append(container)
                    if current[key] is None:
                        next_key = path[i+1] if i+1 < len(path) else None
                        current[key] = [] if isinstance(next_key, int) else {}
                    current = current[key]
                else:  # current is dict
                    if key not in current or current[key] is None:
                        next_key = path[i+1] if i+1 < len(path) else None
                        current[key] = [] if isinstance(next_key, int) else {}
                    current = current[key]
            last_key = path[-1]
            if isinstance(current, list):
                while len(current) <= last_key:
                    current.append(None)
                current[last_key] = value
            else:
                current[last_key] = value
            # Sauvegarder l'état actuel après chaque mise à jour
            self.save_translation(self.current_data, is_temp=True)
        except Exception as e:
            print(f"Erreur lors de la mise à jour du chemin {path}: {e}")
    
    def translate_json_file(self):
        """Traduit le fichier JSON complet"""
        try:
            # Lecture du fichier JSON
            print(f"Lecture du fichier {Config.input_path}...")
            with open(Config.input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Initialiser current_data avec la structure vide
            self.current_data = self.create_empty_structure(data)
            self.create_empty_output_file()
            
            # Traduction récursive
            print("Début de la traduction...")
            translated_data = self.translate_field(data)
            
            # Sauvegarde finale
            self.save_translation(translated_data)
            print(f"Traduction terminée! Résultat sauvegardé dans: {Config.output_path}")
            
            # Suppression du fichier temporaire
            if os.path.exists(Config.temp_path):
                os.remove(Config.temp_path)
            
            return True
            
        except Exception as e:
            print(f"Erreur lors de la traduction: {e}")
            return False
    
    def create_empty_output_file(self):
        """Crée un fichier de sortie vide"""
        try:
            with open(Config.input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            empty_structure = self.create_empty_structure(data)
            
            with open(Config.output_path, 'w', encoding='utf-8') as f:
                json.dump(empty_structure, f, ensure_ascii=False, indent=2)
            
            print(f"Fichier de sortie vide créé: {Config.output_path}")
            
        except Exception as e:
            print(f"Erreur lors de la création du fichier vide: {e}")
    
    def create_empty_structure(self, data):
        """Crée une structure vide basée sur la structure d'origine"""
        if isinstance(data, dict):
            return {k: self.create_empty_structure(v) for k, v in data.items()}
        elif isinstance(data, list):
            return []
        else:
            return None
    
    def save_translation(self, data, is_temp=False):
        """Sauvegarde les données traduites"""
        try:
            output_file = Config.temp_path if is_temp else Config.output_path
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"Erreur lors de la sauvegarde: {e}")
            return False

def main():
    """Fonction principale"""
    print(f"=== Traduction du dataset JSON avec {Config.model_name} ===")
    start_time = time.time()
    
    translator = JsonTranslator()
    success = translator.translate_json_file()
    
    duration = time.time() - start_time
    print(f"Temps d'exécution total: {duration:.2f} secondes ({duration/60:.2f} minutes)")
    
    if success:
        print("Traduction terminée avec succès!")
    else:
        print("La traduction a échoué.")

if __name__ == "__main__":
    main()