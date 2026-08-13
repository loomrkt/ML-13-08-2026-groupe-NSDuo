# ML-13-08-2026-groupe-NSDuo

# **Rapport de Projet — Atlantic Haven Hotels**

## **Examen Final Machine Learning & Data Science — M1**

Réalisé au sein de **ISPM — Madagascar** ([www.ispm-edu.com](https://www.ispm-edu.com))

---

### **1. Informations sur le Groupe**

#### Membre 1

- nom : RAKOTONDRAZAKA 
- prénom(s) : Nameno Fanantenana
- classe : IMTICIA 4
- numéro : 09
- rôle : analyste de données — EDA, interprétation des résultats, rédaction du rapport et présentateur

#### Membre 2

- nom : RAJAONARIVONY
- prénom(s) : Steve Marino
- classe : ISAIA 4
- numéro : 01
- rôle :  responsable de la modélisation — développeur du pipeline ML (prétraitement, feature engineering, entraînement et évaluation des modèles, notebook)
---

### **2. Résumé du Travail**

#### Problématique

Atlantic Haven Hotels subit des annulations de réservations coûteuses : environ **25,8 %** des réservations sont annulées (≈ 1 réservation sur 4), ce qui engendre des chambres vides, des revenus perdus et une planification perturbée (surbooking impossible, coûts de gestion). Une prédiction **suffisamment précoce** du risque d'annulation — dès la création de la réservation — permettrait à l'hôtel de déclencher des actions de fidélisation ou de réallocation avant que la chambre ne soit perdue, avec un coût bien inférieur à l'annulation elle-même.

#### Méthodologie adoptée

1. **EDA** : analyse des valeurs manquantes, de la structure temporelle (réservations créées entre 2023-01 et 2025-05, test entièrement futur) et des corrélations avec la cible (acompte, tarif remboursable, délai réservation→arrivée, canal).
2. **Prétraitement** : imputation médiane (numériques) / modalité la plus fréquente (catégorielles), one-hot encoding robuste (`handle_unknown='ignore'`), standardisation — le tout ajusté **uniquement sur la partie entraînement** pour éviter toute fuite.
3. **Feature engineering** : 15 variables construites (`delai_calc`, prix normalisés par nuit/personne/chambre, `acompte_total`, `remboursable`, `historique_taux_annul`, `attente_ou_modif`, etc.).
4. **Validation temporelle** : entraînement sur les réservations créées avant le **17/10/2024** (6 008 lignes), validation sur celles du **18/10/2024 au 24/05/2025** (1 992 lignes), reproduisant le scénario de déploiement.
5. **Modélisation** : baseline régression logistique sans feature engineering, puis régression logistique + FE, Random Forest, HistGradientBoosting et LightGBM — tous comparés sur le même jeu de validation.
6. **Seuil de décision** : recherche de grille sur le F1 (seuil optimal **0,245**), puis ré-entraînement sur la totalité des données pour la prédiction de test.

#### Résultats obtenus

Meilleur **F1-score de 0,4755** sur la validation temporelle, obtenu avec la **régression logistique régularisée + feature engineering** (précision 0,373 ; rappel 0,656 ; ROC-AUC 0,6583), contre **0,4276** pour la baseline sans feature engineering (gain de **+0,048**). Découverte importante : le **délai d'anticipation**, le **caractère remboursable du tarif** et l'**absence d'acompte** sont les signaux dominants du risque d'annulation — l'acompte total réduit le taux d'annulation de 34 % à 10,4 %.

#### Mots-clés

classification binaire, annulation, validation temporelle, F1-score, feature engineering, seuil de décision, régression logistique, déséquilibre de classes

---

### **3. Contenu du Repository**

Voici la liste des fichiers et liens importants permettant d’évaluer votre travail :

- **notebook.ipynb** : code complet de l’EDA, du prétraitement, de la modélisation et de l’évaluation (51 cellules exécutées) ;
- **submission.csv** : prédictions sur `reservations_test.csv` (2 000 lignes : `reservation_id`, `probabilite_annulation`, `reservation_annulee`) ;
- **README.md** : présent rapport complété ;
- **requirements.txt** : dépendances nécessaires à la reproduction du projet ;
- **ressources/** : jeux de données (`reservations_train.csv`, `reservations_test.csv`, `sample_submission.csv`) et dictionnaire des variables (`data_dictionary.csv`).

**Liens utiles :**

- Lien vidéo de présentation : *(à renseigner)*
- Lien du dépôt GitHub : *(à renseigner)*

---

### **4. Résultats de Modélisation**

Tous les modèles sont évalués sur **le même jeu de validation temporel** (1 992 réservations, période 18/10/2024 → 24/05/2025), le seuil de décision étant optimisé sur ce jeu pour chaque modèle :

| Modèle | Paramètres principaux | F1-score | Précision | Rappel | ROC-AUC |
|---|---|---:|---:|---:|---:|
| Régression logistique — baseline | C=0,1 · solver liblinear · seuil 0,155 · sans feature engineering | 0,4276 | 0,276 | 0,949 | 0,5606 |
| Modèle 2 — Régression logistique + FE | C=0,1 · solver liblinear · seuil 0,245 | 0,4755 | 0,373 | 0,656 | 0,6583 |
| Modèle 3 — Random Forest + FE | 400 arbres · profondeur 14 · feuilles min 4 · seuil 0,285 | 0,4663 | 0,342 | 0,735 | 0,6472 |
| Modèle final — Régression logistique + FE | C=0,1 · solver liblinear · seuil 0,245 | **0,4755** | **0,373** | **0,656** | **0,6583** |

*À titre de comparaison, HistGradientBoosting + FE et LightGBM + FE atteignent respectivement un F1 de 0,4654 et 0,4605 sur le même jeu de validation (seuils 0,160 et 0,100 ; ROC-AUC 0,6355 et 0,6293).*

**Modèle final retenu :** régression logistique régularisée + feature engineering (C=0,1, seuil 0,245).

**Seuil de décision retenu :** **0,245** (maximise le F1 sur la validation temporelle)

**Justification du choix du modèle final :**

Le modèle final est la **régression logistique régularisée enrichie de features métier**, choisie pour quatre raisons :
1. **Performance** : meilleur F1 (0,4755) et meilleur ROC-AUC (0,6583) sur la validation temporelle.
2. **Stabilité** : écart entraînement/validation minimal (F1 train 0,482 vs 0,476), aucun surajustement ; les arbres (RF, HGB, LightGBM) obtiennent un F1 légèrement inférieur malgré leur flexibilité, signe d'une sensibilité à la dérive temporelle.
3. **Interprétabilité** : les coefficients log-odds se traduisent directement en décisions métier (ex. acompte total → −0,33 ; tarif remboursable → +0,32), utilisables par la direction de l'hôtel.
4. **Coût des erreurs** : à son seuil optimal, le modèle équilibre le rappel (0,656 — deux tiers des annulations détectées) et la précision (0,373), cohérent avec un coût des faux négatifs supérieur à celui des faux positifs (voir Q2).

---

### **5. Réponses aux Questions d’Analyse**

#### **Q1. Pourquoi utilise-t-on principalement le F1-score plutôt que l’accuracy pour cette tâche ?**

Parce que la cible est fortement **déséquilibrée** : 74,2 % des réservations ne sont pas annulées contre 25,8 % annulées. Un classifieur naïf prédisant toujours « non annulée » obtiendrait 74,2 % d'accuracy sans détecter **aucune** annulation, ce qui est inutile en pratique. Le F1-score, moyenne harmonique de la précision et du rappel sur la classe minoritaire (annulée), pénalise à la fois les annulations manquées (faux négatifs) et les alertes inutiles (faux positifs) : il mesure la capacité réelle à *trouver* les annulations tout en étant *fiable* dans les alertes. C'est la métrique adaptée quand la classe d'intérêt est minoritaire et que les deux types d'erreurs ont des coûts asymétriques.

#### **Q2. Dans ce contexte, qu’est-ce qui est le plus grave : un faux positif ou un faux négatif ?**

- **Faux positif** : le modèle annonce une annulation qui n'aura pas lieu. Coût : actions de fidélisation ou de surbooking déployées inutilement (budget marketing gaspillé, chambre conservée au lieu d'être revendue), mais aucune perte de revenu directe.
- **Faux négatif** : le modèle ne détecte pas une annulation réelle. Coût : la chambre reste vacante au dernier moment, revenu définitivement perdu, sans possibilité de contre-mesure.

Dans le contexte hôtelier, le **faux négatif est généralement le plus grave** : la perte de revenu est définitive (une chambre non revendue ne se rattrape pas), alors qu'un faux positif coûte peu et peut même être l'occasion d'un geste commercial renforçant la fidélité. C'est pourquoi le seuil retenu (0,245) privilégie le rappel (0,656) au détriment de la précision (0,373) — on préfère 587 faux positifs que 183 annulations manquées. Réponse nuancée : si le budget d'actions de fidélisation est très limité ou si les gestes commerciaux coûtent cher, l'équilibre devrait être recalibré vers plus de précision.

#### **Q3. Quelles variables créées par feature engineering ont le plus amélioré votre modèle par rapport à la régression logistique de référence ?**

Les variables suivantes (construction entre parenthèses) apportent le plus :

- **`delai_calc`** (date_arrivee − date_reservation, en jours) : recalcul exact du délai d'anticipation. Corrélation brute avec la cible la plus forte (0,111) ; coefficient log-odds +0,24. Les réservations faites plus de 180 jours à l'avance annulent à ≈ 57 % contre ≈ 23 % à moins de 7 jours.
- **`remboursable`** (binarisation de `tarif_remboursable`) : +0,32 en coefficient — le tarif remboursable triple presque le taux d'annulation (31,3 % vs 14 %).
- **`acompte_total`** (binarisation de `type_acompte`) : −0,33 — l'acompte total divise par trois le taux d'annulation (10,4 % vs 34 % sans acompte).
- **`historique_taux_annul`** (annulations_passées / (réservations_passées + 1)) : +0,31 — traduit le comportement d'annulation passé du client.
- **`prix_par_nuit`**, **`prix_par_personne_nuit`**, **`prix_par_chambre`** (montant normalisé) : capturent le niveau de prix indépendamment de la durée et du nombre de personnes.

**Gain quantifié : +0,048 en F1** (0,4276 → 0,4755) pour le même algorithme, soit +11,2 % de F1 relatif ; l'AUC progresse de 0,5606 à 0,6583 (+0,098). Les variables construites figurent parmi les 6 premiers coefficients du modèle final.

#### **Q4. Pourquoi un découpage aléatoire simple peut-il produire une évaluation trompeuse sur ce dataset ?**

Parce que le jeu de test est **entièrement dans le futur** : réservations créées entre le 24/05/2025 et le 31/12/2025, alors que l'entraînement couvre 2023-01 → 2025-05. Un découpage aléatoire mélange des réservations passées et futures dans l'entraînement et la validation : le modèle peut mémoriser des **motifs temporels** (saisonnalité, niveaux de prix, comportements de réservation) qui ne seront plus valables à l'inférence — une fuite de données. Notre protocole découpe strictement selon `date_reservation` :

- **Entraînement** : réservations créées avant le **17/10/2024** (6 008 lignes, 75 %) ;
- **Validation** : du **18/10/2024 au 24/05/2025** (1 992 lignes, 25 %, taux d'annulation 26,7 %), juste avant le début du test.

Comparaison chiffrée sur la même régression logistique : F1 = 0,472 (split aléatoire) vs 0,476 (split temporel). L'écart est modeste sur cette fenêtre (signal assez stable), mais le protocole aléatoire reste trompeur *par construction* : il autorise l'apprentissage sur le futur de la validation et peut surestimer fortement les performances dès que la dérive temporelle s'accentue (changements de politique tarifaire, nouvelles campagnes, saisons). La validation temporelle est la seule estimation honnête de la performance de déploiement.

#### **Q5. Quels profils ou scénarios de réservation sont les plus fréquemment associés aux annulations dans vos analyses ?**

- **Réservations souples** : tarif **remboursable** (31,3 % d'annulations vs 14 % non remboursable) et **sans acompte** (34 % vs 23 % partiel, 10,4 % total) — la flexibilité supprime le coût de l'annulation.
- **Très forte anticipation** : délai de réservation > 180 jours (≈ 57 % d'annulations vs 23 % à moins de 7 jours) — les plans lointains sont plus volatils.
- **Réservations via plateforme en ligne** (30,4 %) et segments **groupe** (30,8 %) / **famille** (28,1 %) — vs canaux contrats (entreprise 14,5 %, site hôtel 21,5 %).
- **Séjours modifiés ou passés par la liste d'attente** : la corrélation avec les modifications (0,060) et les combinaisons liste d'attente + modifications signalent une réservation instable.

Attention : ces circonstances décrivent des **interactions observables** (souplesse tarifaire + anticipation + canal), pas des populations intrinsèquement à risque.

#### **Q6. Comment votre pipeline traite-t-il les valeurs manquantes et les catégories jamais observées pendant l’entraînement ?**

- **Valeurs manquantes** : imputation par la **médiane** pour les numériques (`enfants`, `prix_moyen_nuit_eur`, `demandes_speciales`) et par la **modalité la plus fréquente** pour les catégorielles (`marche_origine`). Pour `agent_id`, le manquant est conservé comme modalité à part entière (réservation **directe** sans intermédiaire, ≈ 42 % des lignes). Tous les imputeurs sont ajustés **uniquement sur l'ensemble d'entraînement** (section 4 du notebook) puis appliqués tels quels à la validation et au test — aucune statistique issue du futur ne pollue l'apprentissage (pas de fuite).
- **Catégories jamais observées** : le one-hot encoder est construit avec `handle_unknown='ignore'` et un seuil `min_frequency=15` (les modalités rares sont regroupées dans la catégorie d'ignoration) : une catégorie inconnue à l'inférence est simplement codée par un vecteur de zéros plutôt que de provoquer une erreur ou un artefact. Le pipeline complet (imputation → encodage → scaling) est un `ColumnTransformer` réutilisé par `transform()` pour la validation et le test.

#### **Q7. Selon vous, quelle action l’hôtel devrait-il entreprendre lorsqu’une réservation en cours présente une forte probabilité d’annulation ?**

Une **intervention graduée et non intrusive**, proportionnée au risque et à la valeur du client — jamais l'annulation automatique de la réservation :

1. **Risque élevé (p ≥ 0,245)** : proposer un **geste de sécurisation** — offre de petit-déjeuner, surclassement sous conditions, **tarif partiellement prépayé** avec réduction (convertir une réservation souple en engagement) ; pour les réservations de valeur, un **appel ou email personnalisé** (conseil de séjour, flexibilité affichée).
2. **Risque très élevé (p ≥ 0,4, ≈ 9,4 % des cas)** : **surbooking raisonné** : ajouter la réservation à une liste d'attente de remplacement et, à l'approche de la date, proposer une **reprise de réservation** (report, avoir) pour éviter l'annulation sèche.
3. **Pilotage continu** : mettre à jour le score à J-30, J-14, J-7 (le risque d'annulation décroît à l'approche de l'arrivée) et n'engager les actions payantes qu'à bon escient.

#### **Q8. Votre modèle présente-t-il des performances comparables selon les régions ou les types de destination ?**

Comparaison chiffrée sur la validation (seuil 0,245) :

| Région / destination | n | Taux d'annulation | F1 |
|---|---:|---:|---:|
| Sicilia (insulaire_mixte) | 152 | 38 % | 0.548 |
| Trentino-Alto Adige (montagne) | 185 | 28 % | 0.542 |
| Lombardia (affaires_lacs) | 270 | 27 % | 0.495 |
| Toscana (culturelle_rurale) | 241 | 25 % | 0.384 |
| Sardegna (insulaire_balneaire) | 87 | 26 % | 0.406 |

Les F1 varient de **0,384 à 0,548**. Une partie de cet écart provient du **taux d'annulation réel** (Sicilia, 38 %, offre un meilleur rappel potentiel) ; l'autre partie est un **artefact de taille** : les sous-groupes de moins de ~150 réservations en validation (Sardegna 87, Puglia 115, Liguria 149) produisent des métriques instables — un ou deux cas mal classés font varier le F1 de plusieurs points. Ces limites interdisent de conclure à des différences « intrinsèques » entre régions ; le modèle reste utilisable sur l'ensemble des régions italiennes, avec une confiance moindre sur les petits volumes.

#### **Q9. Analyse des erreurs**

**Cinq faux positifs** (les plus « sûrs » du modèle, probabilité 0,60 à 0,72) : R008802 (délai 452 j, remboursable, prix 113 €/nuit), R003563 (délai 129 j, remboursable, sans acompte), R001349 (délai 429 j, remboursable, 387 €/nuit), R000051 (délai 402 j, remboursable, 2 annulations passées), R000366 (délai 49 j, remboursable). Tous cumulent les signaux statistiques du risque (tarif souple + forte anticipation + prix élevé), mais ces clients ont honoré leur réservation : ce sont des « planificateurs » fiables que le modèle ne distingue pas des annuleurs.

**Cinq faux négatifs** (probabilité 0,24) : R004159 (délai 11 j, tarif non remboursable), R006969 (délai 23 j, 285 €/nuit), R003330 (délai 14 j, 10 nuits), R003471 (délai 21 j, téléphone), R000933 (délai 11 j). Profils « sûrs » : courte anticipation, aucun signal négatif visible — annulations dues à des circonstances **non observables** dans les données (imprévus personnels, changements de plan).

**Raisons possibles :** le signal est modéré (AUC ≈ 0,66) et les variables disponibles décrivent la réservation, pas l'intention du client ; les faux positifs sont les limites d'un modèle qui suit fidèlement des corrélations, les faux négatifs des annulations « exogènes ».

**Piste d'amélioration :** ajouter des données exogènes (événements locaux, météo, prix concurrents, historique client enrichi, comportement de navigation) et surtout modéliser le **temps restant avant l'arrivée** — le risque d'annulation est dynamique ; un score mis à jour à J-30 / J-14 / J-7 (ou une modélisation survival) capturerait une partie de ces cas limites.

---

### **6. Conclusion et Recommandations**

Le modèle retenu (régression logistique + 15 features métier, seuil 0,245) atteint un F1 de **0,4755** et un ROC-AUC de **0,658** sur la période de validation temporelle, soit un gain de +0,048 en F1 sur la baseline sans feature engineering. Ses limites : un signal globalement modéré, des erreurs concentrées sur les réservations remboursables très anticipées (faux positifs) et les annulations de dernière minute sans signe précurseur (faux négatifs), et une fiabilité moindre sur les petits sous-groupes régionaux. Le modèle est utilisable comme **outil d'alerte précoce** (et non de décision automatique), avec un seuil ajustable selon la tolérance au coût et un ré-étalonnage périodique pour suivre la dérive temporelle.

**Recommandation opérationnelle finale :** déployer le score de risque à la création de chaque réservation, déclencher des gestes de sécurisation gradués au-delà du seuil 0,245, re-mesurer le F1 chaque trimestre sur les données fraîches, et enrichir les données avec le statut de paiement et l'échéance avant l'arrivée pour réduire les faux négatifs de dernière minute.

---

### **7. Reproductibilité**

- version de Python : **3.13.4**
- principales bibliothèques et versions : pandas 2.3.2, numpy 2.2.6, scikit-learn 1.8.0, lightgbm 4.7.0, matplotlib 3.10.6, seaborn 0.13.2
- graine(s) aléatoire(s) : **SEED = 42** (tous les modèles et splits)
- commande ou procédure d'exécution : ouvrir `notebook.ipynb` et exécuter toutes les cellules (les fichiers `ressources/*.csv` doivent se trouver dans le dossier `ressources/`). Les prédictions sont écrites dans `submission.csv` par la dernière cellule.
- durée approximative d'entraînement : **≈ 2 à 4 minutes** (machine locale)
- environnement utilisé : **local** (Windows, Python 3.13)

---

### **8. Bibliographie**

- Documentation scikit-learn — Préprocessing, imputation et modèles linéaires : https://scikit-learn.org/stable/
- Documentation LightGBM : https://lightgbm.readthedocs.io/
- Documentation pandas : https://pandas.pydata.org/
- Seaborn — Visualisation de données statistiques : https://seaborn.pydata.org/
- Dictionnaire des variables du concours (`ressources/data_dictionary.csv`) et énoncé (`ressources/readme-model.md`)
- Outils d'IA générative utilisés : assistante IA de développement (classeur notebook, rédaction de sections du rapport) — contribution : aide à la structuration du pipeline et relecture du code ; toutes les analyses, métriques et interprétations ont été produites et vérifiées sur les données réelles.
