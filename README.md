# Asistente inteligente de Investigación de Operaciones

## Instalación (en la terminal de VS Code, dentro de esta carpeta)

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / Mac
pip install -r requirements.txt
```

## Uso

```bash
streamlit run app.py     # interfaz web
python ejemplos.py       # casos de estudio + figuras en /figuras
pytest -v                # pruebas automatizadas (deben pasar 19)
```

## IA opcional

Sin clave todo funciona con el respaldo por reglas. Para activar el modelo de lenguaje:

```bash
set ANTHROPIC_API_KEY=su_clave          # Windows (cmd)
$env:ANTHROPIC_API_KEY="su_clave"       # Windows (PowerShell)
export ANTHROPIC_API_KEY="su_clave"     # Linux / Mac
```
