#!/bin/bash

# Script per pulire i widgets dai notebook Jupyter nella cartella corrente
# Autore: Assistente AI
# Uso: ./clean_jupyter_widgets.sh [opzioni]

# Colori per output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Funzione per mostrare l'aiuto
show_help() {
    echo -e "${BLUE}🧹 JUPYTER WIDGETS CLEANER${NC}"
    echo "=============================="
    echo ""
    echo "Rimuove i widgets dai notebook Jupyter per ridurre le dimensioni dei file."
    echo ""
    echo -e "${YELLOW}Uso:${NC}"
    echo "  ./clean_jupyter_widgets.sh [opzioni]"
    echo ""
    echo -e "${YELLOW}Opzioni:${NC}"
    echo "  -h, --help              Mostra questo aiuto"
    echo "  -r, --recursive         Cerca anche nelle sottocartelle"
    echo "  -b, --no-backup         Non creare file di backup"
    echo "  -v, --verbose           Output dettagliato"
    echo "  -d, --dry-run           Mostra solo cosa verrebbe fatto senza modificare"
    echo "  -f, --force             Non chiedere conferma"
    echo ""
    echo -e "${YELLOW}Esempi:${NC}"
    echo "  ./clean_jupyter_widgets.sh                    # Pulisce i .ipynb nella cartella corrente"
    echo "  ./clean_jupyter_widgets.sh -r -v              # Ricorsivo con output dettagliato"
    echo "  ./clean_jupyter_widgets.sh -d                 # Dry run per vedere cosa accadrebbe"
}

# Variabili di default
RECURSIVE=false
BACKUP=true
VERBOSE=false
DRY_RUN=false
FORCE=false

# Parse degli argomenti
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -r|--recursive)
            RECURSIVE=true
            shift
            ;;
        -b|--no-backup)
            BACKUP=false
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        *)
            echo -e "${RED}❌ Opzione sconosciuta: $1${NC}"
            echo "Usa -h per vedere l'aiuto"
            exit 1
            ;;
    esac
done

# Controlla se jq è installato
if ! command -v jq &> /dev/null; then
    echo -e "${RED}❌ Errore: jq non è installato${NC}"
    echo "Installa jq con:"
    echo "  Ubuntu/Debian: sudo apt install jq"
    echo "  CentOS/RHEL: sudo yum install jq"
    echo "  macOS: brew install jq"
    exit 1
fi

# Trova i file .ipynb
if [ "$RECURSIVE" = true ]; then
    FIND_CMD="find . -name '*.ipynb' -type f"
    SEARCH_DESC="ricorsivamente"
else
    FIND_CMD="find . -maxdepth 1 -name '*.ipynb' -type f"
    SEARCH_DESC="nella cartella corrente"
fi

# Trova tutti i notebook
mapfile -t notebooks < <(eval $FIND_CMD)

if [ ${#notebooks[@]} -eq 0 ]; then
    echo -e "${YELLOW}⚠️  Nessun file .ipynb trovato $SEARCH_DESC${NC}"
    exit 0
fi

echo -e "${BLUE}🔍 Trovati ${#notebooks[@]} notebook $SEARCH_DESC${NC}"

# Analizza ogni notebook
total_size_before=0
total_size_after=0
files_with_widgets=()
files_cleaned=()

for notebook in "${notebooks[@]}"; do
    # Controlla se il file ha widgets
    widgets_count=$(jq '.metadata.widgets."application/vnd.jupyter.widget-state+json" | length // 0' "$notebook" 2>/dev/null)
    
    if [ "$widgets_count" -gt 0 ]; then
        files_with_widgets+=("$notebook:$widgets_count")
        size_before=$(stat -f%z "$notebook" 2>/dev/null || stat -c%s "$notebook" 2>/dev/null)
        total_size_before=$((total_size_before + size_before))
        
        if [ "$VERBOSE" = true ]; then
            echo -e "${YELLOW}📋 $notebook: $widgets_count widgets, $(numfmt --to=iec $size_before)${NC}"
        fi
    fi
done

if [ ${#files_with_widgets[@]} -eq 0 ]; then
    echo -e "${GREEN}✅ Nessun notebook contiene widgets. Tutto pulito!${NC}"
    exit 0
fi

echo ""
echo -e "${YELLOW}📊 RIEPILOGO:${NC}"
echo "  File con widgets: ${#files_with_widgets[@]}"
echo "  Dimensione totale prima: $(numfmt --to=iec $total_size_before)"

# Mostra dettagli se verbose
if [ "$VERBOSE" = true ]; then
    echo ""
    echo -e "${YELLOW}📋 DETTAGLI:${NC}"
    for item in "${files_with_widgets[@]}"; do
        file="${item%:*}"
        count="${item#*:}"
        size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)
        echo "  $file: $count widgets ($(numfmt --to=iec $size))"
    done
fi

echo ""

# Dry run
if [ "$DRY_RUN" = true ]; then
    echo -e "${BLUE}🔍 DRY RUN - Nessun file verrà modificato${NC}"
    for item in "${files_with_widgets[@]}"; do
        file="${item%:*}"
        count="${item#*:}"
        echo -e "  ${YELLOW}Pulirebbe:${NC} $file ($count widgets)"
    done
    exit 0
fi

# Chiedi conferma se non in modalità force
if [ "$FORCE" = false ]; then
    echo -e "${YELLOW}❓ Vuoi procedere con la pulizia? (y/N)${NC}"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}🚫 Operazione annullata${NC}"
        exit 0
    fi
fi

echo ""
echo -e "${BLUE}🧹 Inizio pulizia...${NC}"

# Pulisci ogni file
for item in "${files_with_widgets[@]}"; do
    file="${item%:*}"
    count="${item#*:}"
    
    # Backup se richiesto
    if [ "$BACKUP" = true ]; then
        backup_file="${file}.backup.$(date +%Y%m%d_%H%M%S)"
        cp "$file" "$backup_file"
        if [ "$VERBOSE" = true ]; then
            echo -e "  ${BLUE}💾 Backup:${NC} $backup_file"
        fi
    fi
    
    # Rimuovi widgets
    temp_file=$(mktemp)
    if jq 'del(.metadata.widgets)' "$file" > "$temp_file" 2>/dev/null; then
        mv "$temp_file" "$file"
        
        size_after=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)
        total_size_after=$((total_size_after + size_after))
        files_cleaned+=("$file")
        
        if [ "$VERBOSE" = true ]; then
            echo -e "  ${GREEN}✅ Pulito:${NC} $file ($count widgets rimossi)"
        fi
    else
        echo -e "  ${RED}❌ Errore:${NC} $file"
        rm -f "$temp_file"
    fi
done

# Riepilogo finale
echo ""
echo -e "${GREEN}🎉 PULIZIA COMPLETATA!${NC}"
echo "=============================="
echo "  File processati: ${#files_cleaned[@]}"
echo "  Dimensione prima: $(numfmt --to=iec $total_size_before)"
echo "  Dimensione dopo: $(numfmt --to=iec $total_size_after)"

if [ $total_size_before -gt 0 ]; then
    reduction=$((total_size_before - total_size_after))
    percentage=$((reduction * 100 / total_size_before))
    echo "  Spazio risparmiato: $(numfmt --to=iec $reduction) (${percentage}%)"
fi

if [ "$BACKUP" = true ]; then
    echo ""
    echo -e "${BLUE}💾 File di backup creati con suffisso .backup.YYYYMMDD_HHMMSS${NC}"
    echo -e "${YELLOW}   Rimuovi i backup quando sei sicuro: rm *.backup.*${NC}"
fi

echo ""
echo -e "${GREEN}✨ Tutti i widgets sono stati rimossi con successo!${NC}"