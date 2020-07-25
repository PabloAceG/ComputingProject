1. Simplificar:
- Usar metricas de complejidad para ver si están correlacionadas con el 
balanceo (resultado de clasificación con el balanceo).
- Sacar métricas de complejidad y ver como están relacionadas.
- A mayor overlapping peor clasifica.

2. Extensión:
- Por donde generar los datos para minimizar el overlapping.

3.
- Invalance-Learn - Python -> Métricas desbalanceadas
- DCol - C++ -> Mide medidas de complejidad
- Reticulate - R -> Ejecutar Python en R

4.
- Mandar correo mario: librería de Python

---

29/04/20

1. 
- Datasets ya hecho
  - Coger los que pone OO o los de Eclipse 
- Pensar en un dataset sintético

2.
- Usar validación cruzada, ver como afecta el usarlo con los diferentes folds
  - Ver como varía del global (sin k-fold)
  - Mirar también los fallos
    - Si el dataset es muy pequeño
	
3.
- !!! Usar Jupiter

---

02/02/20

1. Además se puede echar un ojo a estas métricas para el desbalanceo 
(y papers asociados):

https://github.com/jonathanSS/ImbalanceDegree

---

10/06/200

- Explicar código, no intro de proyecto
- (EMPEZAR POR ESTO)
  - Centrarse principalmente en desbalanceo (como métricas afectan a desbalanceo)
    - Lanzar clasificador "scikit-learn" (con los de defectos preferiblemente - 
      es insistente con ello)
      - https://scikit-learn.org/stable/supervised_learning.html#supervised-learning
      - Recomiemda Naive Bayes
- Como afectan las métricas - ver correlación entre métricas entre lo bien o 
  mal que se puede clasificar ("alguno de estos de 4.5")
- Con cross-val de 10 
  - Mirar si los datasets intermedios tienen afectadas las métricas ("si se 
    mantienen las métricas)
  - Usar ejemplo "Rites" - enlace de arriba
  - (FINAL) Comparar y evaluar 
    - https://sherbold.github.io/autorank/
  - Cross project defect prediction
	  - https://github.com/Riteshgoenka/SoftwareDefectPrediction
	  - Con un proyecto intentar predecir otros defectos en otro proyecto
	  - https://www.overleaf.com/project/5db2b65ed0080f0001019a29

---

8/7/2020

- Project proposal
  - Use the project proposal from Brookes
  - Sign paperwork
    - Anexo 3 ... Anteproyecto
    - https://escuelapolitecnica.uah.es/estudiantes/trabajo-fin-grado.asp
      - Fill in and sign page 17
- Read .arff classes
  - Must have binary target (yes or no)
    - O transform it: 
      - == 0 -> No
      - > 0 -> Yes
  - Delete string columns (no need of class, version, etc)
    - They all have either same values or different values (also useless)
- Use Ritesh final project to know how to read files
	  