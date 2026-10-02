# Fichiers officiels à déposer ici

Ces fichiers ne sont pas versionnés : ils appartiennent à leurs auteurs et doivent être
téléchargés directement à la source.

| Fichier attendu | Où le trouver | À quoi il sert |
|---|---|---|
| `EIOPA_RFR_20251231_Term_Structures.xlsx` | site de l'EIOPA, rubrique « Risk-free interest rate term structures » | courbe des taux au 31/12/2025 et facteurs officiels de choc |
| classeur TH / TF 00-02 | tables réglementaires françaises | mortalité des garanties décès |
| classeur TGH05 / TGF05 | tables réglementaires françaises | mortalité des rentes, en générationnel |

Une fois les fichiers déposés, lancer les deux scripts d'extraction :

```bash
python scripts/extraire_parametres_eiopa.py sources/EIOPA_RFR_20251231_Term_Structures.xlsx
python scripts/extraire_tables_mortalite.py sources/TH-TF-00-02.xls sources/TGF05-TGH05.xls 2025
```

Sans ces fichiers, le projet reste exécutable : `courbe.source` bascule sur `smith_wilson` et
`mortalite.source` sur `makeham`, deux approximations documentées dans le rapport.
