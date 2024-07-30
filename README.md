# IA-AprendizajePorRefuerzo
**AprendizajeRefuerzo** es un proyecto realizada para la asignatura de IA, de Ingeniería del Software en la Universidad de Sevilla. En este trabajo nos centraremos en aplicar el aprendizaje por refuerzo a la robótica móvil, donde un robot debe planificar una ruta en un entorno con obstáculos.

## DEFINICIÓN DEL PROBLEMA

Suponemos un mapa con obstáculos representado por una matriz binaria, donde 0 representará un celda libre y un 1 la presencia de un obstáculo. Cada celda de la matriz se corresponderá con una superficie de 0.5 x 0.5 metros.

El robot puede localizarse en cualquier celda libre de obstáculos y podrá desplazarse a cualquiera de las 8 celdas subyacentes, siempre que esté libre de obstáculos. Considerando una celda cono Pos. inicial, $P_{init}$, y una celda como posición final $P_{end}$, podemos definir el problema de la siiguiente manera:

- **Matriz de ocupacion $O$ de $n$ filas por $m$ columnas**: Cada celda de la matriz contiene un 1 si la posición está ocupada por obstáculo o un 0 si está libre de obstáculos. Para evitar que el robot se salga del mapa, todo el contorno del mapa serán obstáculos. Es decir *la primera como la última fila de la matriz y la primera y última columna de la matriz tendrán el valor 1*.

- **Posicion inicial $P_{init}$ = ($x_a$, $y_a$) y posición final $P_{end}$ = ($x_b$, $y_b$)**: Con 0 <= $x_a$, $x_b$ < $n$ y 0 <= $y_a$, $y_b$ < $m$. Ambas posiciones deben ser libres de obstáculos. La restricción con comparativos nos quiere indicar que las posiciones deben estar dentro de los limites del mapa.

- **Conjunto de estados $S$** = {($x$, $y$) | 0 <= $x$ < $n$ y 0 <= $y$ < $m$}. Son todas las posibles posiciones que ofrece el mapa y en las que se puede encontrar el robot. Es cierto que estos estados también incluyen celdas con obstáculos. Por definición no se impide que el robot se coloque en estas celdas con obstáculos, pero estará fuertemente penalizado cuando lo haga por la función de recompensa.

- **Conjunto de acciones A**: {N,NE,E,SE,S,SO,O,NO,esperar} que representan los movimientos en las 8 posibles direcciones. La acción *esperar* se utiliza para indicar que el robot no se mueve, y se queda en la misma posición.

- **Función de transición $T$**: $S$ x $A$ $\rightarrow$ $S$. Esta función nos indica el estado resultante al aplicar una acción. Esta acción será estocástica (no determinista, "aleatoria"), ya que modelizará un cierto error en la ejecución del movimiento y viene dado por $T(s,a)$ = $s'$ donde:
    - $s$ es el estado de partida.
    - $a$ es la acción a realizar.
    - $s'$ es el estado resultante después de realizar la acción $a$ en el estado $s$.
    - Si $s$ contiene obstáculos, entonces $s'$ = $s$ independientemente de la acción $a$. Si estamos sobre un obstáculo, no podemos movernos.
    - Si $a$ = $esperar$, entonces $s'$ = $s$. El efecto de dicha acción es determinista
    - Para el resto de acciones, si $s$ es libre de obstáculos, con una probabilidad de $1 - P_{error}$ se producirá el movimiento deseado. Dado $s$ = ($x$, $y$), vendrá definido por:
        -  Si $a$ = $N$, entonces $s'$ = ($x$, $y-1$).
        -  Si $a$ = $NE$, entonces $s'$ = ($x+1$, $y-1$).
        -  Si $a$ = $E$, entonces $s'$ = ($x+1$, $y$).
        -  Si $a$ = $SE$, entonces $s'$ = ($x+1$, $y+1$).
        -  Si $a$ = $S$, entonces $s'$ = ($x$, $y+1$).
        -  Si $a$ = $SO$, entonces $s'$ = ($x-1$, $y+1$).
        -  Si $a$ = $O$, entonces $s'$ = ($x-1$, $y$).
        -  Si $a$ = $NO$, entonces $s'$ = ($x-1$, $y-1$).
    - Por otra parte, con una probabilidad $P_{error}$, se producirá un movimiento de error $a'$ que vendrá dado con una desviación o a la izquierda o a la derecha sobre la dirección deseada, de forma equiprobable.
        - Si $a$ = $N$, $a'$ podrá ser $NE$ o $NO$, con una probabilidad de $P_{error}/2$ cada uno. (Equiprobable)
        - Si $a$ = $NE$, $a'$ podrá ser $N$ o $E$, con una probabilidad de $P_{error}/2$ cada uno.
        - Si $a$ = $E$, $a'$ podrá ser $NE$ o $SE$, con una probabilidad de $P_{error}/2$ cada uno.
        - Si $a$ = $SE$, $a'$ podrá ser $E$ o $S$, con una probabilidad de $P_{error}/2$ cada uno.
        - Si $a$ = $S$, $a'$ podrá ser $SE$ o $SO$, con una probabilidad de $P_{error}/2$ cada uno.
        - Si $a$ = $SO$, $a'$ podrá ser $E$ o $O$, con una probabilidad de $P_{error}/2$ cada uno.
        - Si $a$ = $O$, $a'$ podrá ser $SO$ o $NO$, con una probabilidad de $P_{error}/2$ cada uno.
        - Si $a$ = $NO$, $a'$ podrá ser $N$ o $O$, con una probabilidad de $P_{error}/2$ cada uno.

    - **Función de recompensa** $R(s,a)$ tiene el siguiente valor:
        - $R(s,a)$ = $K1$ si $s$ ≠ $P_{end}$ y $a$ = $esperar$.
        - $R(s,a)$ = $R(s)$ en cualquier otro caso.
        - $R(s)$ = $K2$ si $s$ es un obstáculo (Si la casilla tiene un 1 en la matriz).
        - $R(s)$ = - $d$ si el estado $s$ no es un obstáculo, siendo $d$ la distancia Euclídea desde $s$ hasta el destino $P_{end}$.

        Los valores $K1$ y $K2$ deberán ser negativos. Se recibirá una penalización alta ($K2$) si el robot está en un obstáculo, y -$d$ que será menor cuanto más cerca estemos del destino. La penalización $K1$ es para que el robot no se quede esperando en una posición que no sea el destino. Usaremos $K1$ = -100, $K2$ = -1000

## ALGORITMOS UTILIZADOS

Esta proyecto proporciona un marco para la resolución de este **problema** y la obteción de políticas usando los siguientes algoritmos:

- Q-Learning y Monte-Carlo. (Temas de la asignatura)
- Algoritmo SARSA [SARSA](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)
