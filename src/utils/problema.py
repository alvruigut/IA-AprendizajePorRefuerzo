import numpy as np
import matplotlib.pyplot as plt

class Problema:
    
    #Constructor
    def __init__(self, mapa, prob_error):
        self.mapa, self.destino = self.lee_mapa(mapa)
        self.estados = self.estados(self.mapa)
        self.acciones = ['esperar', 'N', 'NE', 'E', 'SE', 'S', 'SO', 'O', 'NO']
        self.recompensas = self.crea_recompensas_sistema()
        self.transiciones = self.crea_transiciones_sistema(prob_error)
        self.politica = self.crea_politica_greedy(self.estados, self.acciones, self.mapa, self.destino)

    '''
    El siguiente código genera la matriz de recompensas SxA, en dónde:

    * R(s,a) = -100 si *s* es un estado distinto del goal y *a* es *esperar*
    * R(s,a) = R(s) en cualquier otro caso
    
    '''
    def crea_recompensas_sistema(self):
        matriz = []
        for e in self.estados:
            fila = [self.obtiene_recompensa(e, self.destino, self.mapa) for _ in self.acciones]
            if e != self.destino:
                fila[0] = -100
            matriz.append(fila)
        return np.array(matriz)

# El siguiente código genera todas las transiciones del sistema como un array AxSxS
    def crea_transiciones_sistema(self, prob_error):
        return np.array([self.crea_transiciones_movimiento(accion, prob_error, self.estados, self.mapa) for accion in self.acciones])
        
        #Actualiza la politica
    def politica_actualizar(self, politica):
        self.politica = politica
    
   # Situa al robot en una posición inicial para simular su movimiento 
    def posicion_inicial(self):
        estado_inicial = self.generar_estado_aleatorio()
        while self.hay_colision(estado_inicial, self.mapa):
            estado_inicial = self.generar_estado_aleatorio()
        return estado_inicial

    # Genera estados dentro del rango del mapa de forma aleatoria
    def generar_estado_aleatorio(self):
        x = np.random.randint(0, self.mapa.shape[1])
        y = np.random.randint(0, self.mapa.shape[0])
        return (x, y)
            
   #  Visualiza el movimiento del robot y muestra finalmente su recorrido
    def visualizar_movimiento(self, estado_inicial):
        estado_actual = estado_inicial
        camino = [estado_actual]

        while estado_actual != self.destino:
            # Obtén la acción basada en la política
            indice_estado = self.obtiene_indice_estado(estado_actual, self.mapa)
            accion = self.politica[indice_estado]
            # Aplica la acción para obtener el nuevo estado
            nuevo_estado = self.aplica_accion(estado_actual, accion, self.mapa)
            if nuevo_estado in camino:
                print("Has entrado en un bucle infinito")
                break
            # Verifica si hay colisión
            if self.hay_colision(nuevo_estado, self.mapa):
                print("No tienes más movimientos")
                break
            # Actualiza el estado actual
            estado_actual = nuevo_estado
            camino.append(estado_actual)
            # Verifica si ha llegado al destino
            if estado_actual == self.destino:
                print("Se ha llegado al destino.")
                break
        self.visualiza_mapa(self.mapa, self.destino)
        for estado in camino:
            plt.plot(estado[0], estado[1], 'bo', markersize=20)   
        plt.show()

        return f'Camino seguido por el agente: {camino}'
    

    # Métodos estáticos para funciones auxiliares

    # El siguiente código lee el goal(destino) y el mapa del fichero
    @staticmethod
    def lee_mapa(fichero):
        with open(fichero,'r') as archivo:
            lineas = archivo.readlines()
        numeros = [float(numero) for numero in lineas[0].split()]
        lineas.pop(0)
        lineas.reverse()
        matriz = []
        for linea in lineas:
            fila = [int(caracter) for caracter in linea.strip()]
            matriz.append(fila)
        return np.array(matriz), (numeros[0], numeros[1])

    # Función para visualizar el mapa
    @staticmethod
    def visualiza_mapa(mapa, destino):
        plt.figure(figsize=(len(mapa[0]), len(mapa)))
        plt.imshow(1 - mapa, cmap='gray', interpolation='none')
        plt.xlim(-0.5, len(mapa[0]) - 0.5)
        plt.ylim(-0.5, len(mapa) - 0.5)
        plt.gca().add_patch(plt.Rectangle((destino[0] - 0.5, destino[1] - 0.5), 1, 1, edgecolor='black', facecolor='red', lw=5))

   # Genera una lista de estados 
    @staticmethod
    def estados(mapa):
        estados = []
        for i in range(0, mapa.shape[1]):
            for j in range(0, mapa.shape[0]):
                estados.append(tuple([i, j]))
        return estados

    # Comprueba si un estado está en un obstáculo
    @staticmethod
    def hay_colision(estado, mapa):
        return mapa[estado[1], estado[0]] == 1

    @staticmethod
    def aplica_accion(estado, accion, mapa):
        if Problema.hay_colision(estado, mapa):
            return estado
        x = estado[0]
        y = estado[1]

        if accion == 'N':
            y += 1
        elif accion == 'S':
            y -= 1
        elif accion == 'E':
            x += 1
        elif accion == 'O':
            x -= 1
        elif accion == 'NE':
            y += 1
            x += 1
        elif accion == 'SE':
            y -= 1
            x += 1
        elif accion == 'SO':
            y -= 1
            x -= 1
        elif accion == 'NO':
            y += 1
            x -= 1
        return x, y

    '''
La recompensa en un estado es:

* -K si el estado es un obstáculo
* -distancia_euclidea_al_goal si el estado no es un obstáculo

Penalizamos los obstáculos y en cualquier otro caso, habrá recompensa menos negativa cuanto más cerca se esté del objetivo
    '''
    @staticmethod
    def obtiene_recompensa(estado, destino, mapa):
        K2 = -1000
        if Problema.hay_colision(estado, mapa):
            valor = K2
        else:
            d = np.sqrt((estado[0] - destino[0]) ** 2 + (estado[1] - destino[1]) ** 2)
            valor = -d
        return valor

# El siguiente código visualiza la recompensa de los estados libres de obstáculos con un gradiente de color desde blanco (menor recompensa) a azul (mayor recompensa)
    @staticmethod
    def visualiza_recompensas(mapa, destino):
        Problema.visualiza_mapa(mapa, destino)
        mapa_estados = Problema.estados(mapa)
        recompensas = [Problema.obtiene_recompensa(e, destino, mapa) for e in mapa_estados]
        recompensas = [np.nan if elemento == -1000 else elemento for elemento in recompensas]
        max_recompensa = np.nanmax(recompensas)
        min_recompensa = np.nanmin(recompensas)
        for e in mapa_estados:
            r = Problema.obtiene_recompensa(e, destino, mapa)
            if r == -1000:
                continue
            a = (r - min_recompensa) / (max_recompensa - min_recompensa)
            rect = plt.Rectangle((e[0] - 0.5, e[1] - 0.5), 1, 1, alpha=a, linewidth=1, edgecolor='blue', facecolor='blue')
            plt.gca().add_patch(rect)


    '''
    Vamos a suponer que el movimiento del robot no es perfecto y puede desviarse a izquierda o derecha de su trayectoria, es decir:

* Si va en dirección N, podría terminar en dirección NE o NO
* Si va en dirección S, podría terminar en dirección SE o SO
* Si va en dirección E, podría terminar en dirección NE o SE
* Si va en dirección O, podría terminar en direcicón NO o SO
* Si va en dirección NE, podría terminar en dirección N o E
* Si va en dirección NO, podría terminar en dirección N u O
* Si va en dirección SE, podría terminar en dirección S o E
* Si va en dirección SO, podría terminar en dirección S u O

El siguiente código devuelve la lista de posibles acciones de error dada una acción. Téngase en cuenta, que la acción esperar no tiene error.
    '''
    @staticmethod
    def obtiene_posibles_errores(accion):
        if accion == 'N':
            errores = ['NE', 'NO']
        elif accion == 'S':
            errores = ['SE', 'SO']
        elif accion == 'E':
            errores = ['NE', 'SE']
        elif accion == 'O':
            errores = ['NO', 'SO']
        elif accion == 'NE':
            errores = ['N', 'E']
        elif accion == 'NO':
            errores = ['N', 'O']
        elif accion == 'SE':
            errores = ['S', 'E']
        elif accion == 'SO':
            errores = ['S', 'O']
        else:
            errores = []
        return errores

    '''
    Dado un estado en forma de tupla, devuelve el indice del estado en el mapa
    '''
    @staticmethod
    def obtiene_indice_estado(estado, mapa):
        return int(estado[0] * mapa.shape[0] + estado[1])
    
    '''
    Dado el indice de un estado, devuelve la tupla del estado
    '''
    @staticmethod
    def obtiene_estado_desde_indice(indice, mapa):
        y = indice % mapa.shape[0]
        x = indice // mapa.shape[0]
        return (x+0.0, y+0.0)

    '''
    El siguiente código crea la matriz de transición SxS para una acción determinada y una probabilidad de error *prob_error*

* Con probabilidad *1 - prob_error*, el estado resultante será el correspondiente a aplicar la acción correctamente.
* Con probabilidad *prob_error/N*, el estado resultante será el correspondiente a aplicar una de las posibles acciones de error devueltas por la función anterior, en donde N es el número de acciones de error
    '''
    @staticmethod
    def crea_transiciones_movimiento(accion, prob_error, estados, mapa):
        matriz = []
        for e0 in estados:
            fila = [0] * len(estados)
            if Problema.hay_colision(e0, mapa):
                fila[Problema.obtiene_indice_estado(e0, mapa)] = 1
            else:
                goal = Problema.aplica_accion(e0, accion, mapa)
                errores = Problema.obtiene_posibles_errores(accion)
                if len(errores) == 0:
                    fila[Problema.obtiene_indice_estado(goal, mapa)] = 1
                else:
                    fila[Problema.obtiene_indice_estado(goal, mapa)] = 1 - prob_error
                    for error in errores:
                        goal_error = Problema.aplica_accion(e0, error, mapa)
                        fila[Problema.obtiene_indice_estado(goal_error, mapa)] = prob_error / len(errores)
            matriz.append(fila)
        return np.array(matriz)

    
    # Función que visualiza la política de un problema  
    @staticmethod
    def visualiza_politica(politica, mapa, destino, estados):
        Problema.visualiza_mapa(mapa, destino)
        for p in zip(estados, politica):
            accion = p[1]
            if accion == 'esperar':
                continue
            estado = p[0]
            e1 = Problema.aplica_accion(estado, accion, mapa)
            x0 = estado[0]
            y0 = estado[1]
            x1 = e1[0]
            y1 = e1[1]

            plt.gca().arrow(x0, y0, (x1 - x0) * 0.6, (y1 - y0) * 0.6,
                            head_width=0.3, head_length=0.3, fc='black', ec='black')

    '''
Política ambiciosa o *greedy* que asigna a cada estado la acción que más acercaría el robot al objetivo (sin colisionar).

El siguiente código calcula dicha política y la devuelve como una lista de acciones, una por cada estado.
    '''
    @staticmethod
    def crea_politica_greedy(estados, acciones, mapa, destino):
        p = []
        for e in estados:
            valores = []
            for a in acciones:
                e1 = Problema.aplica_accion(e, a, mapa)
                valores.append(Problema.obtiene_recompensa(e1, destino, mapa))
            accion = acciones[np.argmax(valores)]
            p.append(accion)
        return p
