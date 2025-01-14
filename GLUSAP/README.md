# **GLUSAP - Dashboard per a la Predicció de Glucosa**

GLUSAP és una aplicació interactiva desenvolupada amb Python i Dash per monitoritzar, simular, i predir nivells de glucosa en pacients diabètics utilitzant dades reals i simulades.

## **Funcionalitats**
- **Predicció de Glucosa**: Visualitza les prediccions generades per models LSTM per a pacients específics.
- **Simulacions**: Explora simulacions de nivells de glucosa en diferents categories (heterogènies, homogènies baixes i altes).
- **Dades Reals**: Analitza dades reals de glucosa provinents del dataset OhioT1DM amb opcions per visualitzar dades processades i no processades.

---

## **Instal·lació**

### **Prerequisits**
- Python 3.10 o superior
- Llibreries requerides (indicat a `poetry.lock`):

### **Pasos per configurar**
1. Clona el repositori:
   ```bash
   git clone https://github.com/danaecanillas/GLUSAP.git
   cd GLUSAP
   ```

### **Accés a les Dades**
Per accedir a les dades utilitzades pel projecte (tant les dades simulades com les reals), cal contactar amb l'autora:
Dànae Canillas

### **Estructura del projecte**
  ```plaintext
GLUSAP/
├── data/                      # Conté dades reals i processades
├── src/                       # Codi font
│   ├── models/                  
│   ├── data_loader.py                 
│   ├── main.py                
│   ├── report_dades.py                
│   ├── report_dades_reals.py                 
│   ├── simulation.py                 
│   └── app.py                 # Fitxer principal de l'aplicació Dash
├── README.md                  # Documentació del projecte
  ```

### **Com Utilitzar**
Executa l'aplicació:
```bash
   python src/app.py
```
Accedeix al dashboard mitjançant el navegador a http://127.0.0.1:8050.
