# Análisis Numérico – Métodos Computacionales (Interfaz Gráfica)

Este proyecto es una aplicación de escritorio en **Python** con interfaz gráfica en **Tkinter** que integra los principales métodos de **análisis numérico** vistos en el curso, organizados por capítulos:

- **Capítulo 1:** Métodos de búsqueda de raíces.
- **Capítulo 2:** Métodos iterativos para sistemas de ecuaciones lineales.
- **Capítulo 3:** Métodos de interpolación.

La aplicación permite:
- Ingresar datos de forma guiada.
- Ver tablas de resultados en la interfaz.
- Graficar funciones y aproximaciones (en algunos métodos).
- Generar informes y comparar métodos según diferentes tipos de error.

---

## 👥 Integrantes

- Mateo Cadavid Ramirez - mcadavidr2
- Daniela Alvarez - dalvarez9

---

## ⚙️ Requisitos del sistema

Antes de ejecutar la aplicación, asegúrate de contar con lo siguiente:

- **Sistema operativo**:  
  - Windows 10/11, Linux o macOS.
- **Python**:
  - Versión **3.9 o superior** instalada y configurada en la variable de entorno `PATH`.
- **Tkinter**:
  - Windows: normalmente viene incluido con la instalación estándar de Python.
  - Linux (ejemplo Debian/Ubuntu): puede requerir instalación manual:
    ```bash
    sudo apt-get update
    sudo apt-get install python3-tk
    ```
- **Git** (opcional pero recomendado) para clonar el repositorio.
- Conexión a internet para instalar las dependencias con `pip`.

---

## 📥 Clonación del repositorio

Si tienes **Git** instalado, puedes clonar el proyecto con:

```bash
git clone https://github.com/mcadavidr2/AnalisisNumericoProyecto.git
```
---

## Si no usas Git, también puedes:

1. Entrar al repositorio en la web.

2. Descargarlo como ZIP.

3. Extraerlo y abrir la carpeta del proyecto en tu PC.

---

## 🧪 Creación y activación del entorno virtual (opcional pero recomendado)

🔹 En Windows

```bash
python -m venv venv
venv\Scripts\activate
```

🔹 En Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```
---

## 📦 Instalación de dependencias

🔹 En Windows

```bash
pip install -r requirements.txt
```

🔹 En Linux / macOS

```bash
pip3 install -r requirements.txt
```

---

## 📦 requirements.txt

El archivo **`requirements.txt`** cuenta con este contenido:

```txt
numpy
pandas
matplotlib
tabulate
```
---

## 📚 Descripción por capítulos

### 🔹 Capítulo 1 – Ecuaciones No Lineales

Métodos implementados:

- **Bisección**
- **Regla Falsa**
- **Punto Fijo**
- **Newton**
- **Secante**
- **Raíces Múltiples** (una de raíces múltiples)

Funcionalidades:

- Ingreso de:
  - Función `f(x)` (y/o `g(x)` en punto fijo).
  - Intervalos o valores iniciales.
  - Tolerancia.
  - Número máximo de iteraciones.
- Muestra:
  - **Tabla de iteraciones** en la interfaz.
  - **Gráfica de la función y aproximaciones** (para métodos de raíces seleccionados).
- Apoyo al usuario:
  - Ejemplos de funciones.
  - Explicación de cómo ingresar `f(x)` en sintaxis de Python (`x**2 - 2`, `np.sin(x)`, etc.).
- Informes:
  - Informe de ejecución y comparación entre métodos para un error específico:
    - Error **relativo**, **absoluto** o **de condición**.
  - Identificación del **mejor método** según el criterio seleccionado.
  - El informe puede activarse o no según la elección del usuario.

---

### 🔹 Capítulo 2 – Sistemas de Ecuaciones Lineales

Métodos implementados:

- **Jacobi**
- **Gauss-Seidel**
- **SOR** (Successive Over-Relaxation)

Funcionalidades:

- Ingreso de:
  - Matriz **A** (hasta tamaño **7×7**, filas separadas por `;` y columnas por `,`).
  - Vector **b**.
  - Vector inicial `x0`.
  - Tolerancia y número máximo de iteraciones.
  - Factor de relajación `w` (en SOR).
- Muestra:
  - Tabla de iteraciones en la interfaz.
  - **Radio espectral** y verificación de convergencia.
  - Mensaje indicando si el método **puede o no converger**.
- Informes:
  - Informe de ejecución y comparación entre Jacobi, Gauss-Seidel y SOR.
  - Comparación según diferentes errores.
  - Identificación del mejor método.

---

### 🔹 Capítulo 3 – Interpolación

Métodos implementados:

- **Vandermonde**
- **Newton Interpolante**
- **Lagrange**
- **Spline Lineal**
- **Spline Cúbico**

Funcionalidades:

- Ingreso de:
  - Puntos `x` y `y` (hasta **8 datos**).
- Muestra:
  - Polinomio de interpolación (o polinomios por tramo).
  - Posibilidad de ver los polinomios completos en una ventana modal.
  - Gráfica de la interpolación (según implementación de los módulos).
- Informes:
  - Comparación entre métodos en términos de errores.
  - Identificación del mejor método para el problema dado.
- Ayuda:
  - Explicación de cómo ingresar los datos (`x1,x2,x3,...`, `y1,y2,y3,...`).

---
