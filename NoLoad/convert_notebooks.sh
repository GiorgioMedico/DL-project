#!/bin/bash

# Script per convertire tutti i file Jupyter Notebook in formato Markdown
# Utilizza: ./convert_notebooks.sh

echo "Avvio conversione dei file Jupyter Notebook nella directory corrente..."

# Verifica se jupyter nbconvert è disponibile
if ! command -v jupyter &> /dev/null; then
    echo "Errore: jupyter non è installato o non disponibile nel PATH"
    exit 1
fi

# Contatore per tenere traccia dei file processati
converted_files=0
error_files=0

# Elabora tutti i file .ipynb nella directory corrente
for notebook in *.ipynb; do
    # Verifica se esistono file .ipynb
    if [ ! -f "$notebook" ]; then
        echo "Nessun file .ipynb trovato nella directory corrente"
        exit 0
    fi
    
    # Estrae il nome base del file senza estensione
    basename=$(basename "$notebook" .ipynb)
    output_file="${basename}.md"
    
    echo "Conversione in corso: $notebook -> $output_file"
    
    # Esegue la conversione utilizzando nbconvert
    if jupyter nbconvert --to markdown "$notebook" --output "$basename.md"; then
        echo "Conversione completata: $output_file"
        ((converted_files++))
    else
        echo "Errore durante la conversione di: $notebook"
        ((error_files++))
    fi
    
    echo "----------------------------------------"
done

# Riepilogo finale
echo "Conversione terminata:"
echo "File convertiti con successo: $converted_files"
echo "File con errori: $error_files"

if [ $converted_files -gt 0 ]; then
    echo "I file convertiti sono disponibili nella directory corrente con estensione .md"
fi