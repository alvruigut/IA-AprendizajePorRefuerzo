import numpy as np
import random
from utils import problema as util

'''
    Clase que implementa el algoritmo de Q-Learning para la resolución de problemas de aprendizaje por refuerzo.
        Q[estado][accion] += alpha * (recompensa + gamma * np.max(Q[siguiente_estado]) - Q[estado][accion])
'''

class QLearning:
    
    # Constructor
    def __init__(self, transiciones, recompensas, gamma=0.9, alpha=0.25, epsilon=0.5, max_iteraciones=2000, max_pasos=100):
        '''
        ################################################### PARÁMETROS ######################################################################
            transiciones: 
                Type: List
                Description: Matriz de probabilidades de transición. Cada fila representa un estado y cada columna una acción, y su valor la probabilidad de transitar al siguiente estado.

            recompensas: 
                Type: List
                Description: Matriz de recompensas. Cada fila representa un estado y cada columna una acción y su valor la recompensa asociada a ambas.

            gamma: 
                Type: float
                Description: Factor de descuento. DefaultValue=0.9

            alpha: 
                Type: float
                Description: Factor de aprendizaje. DefaultValue=0.25

            max_iteraciones: 
                Type: int
                Description: Número máximo de iteraciones. DefaultValue=2000

            max_pasos: 
                Type: int
                Description: Número de pasos en cada iteración. DefaultValue=100

            epsilon: 
                Type: float
                Description: Probabilidad de exploración. DefaultValue=0.5
        '''
        self.transiciones = transiciones
        self.recompensas = recompensas
        self.gamma = gamma  # Factor de descuento
        assert 0.0 < self.gamma <= 1.0, "El factor de descuento debe estar entre 0 y 1"
        self.alpha = alpha  # Factor de aprendizaje
        assert 0.0 < self.alpha <= 1.0, "El factor de aprendizaje debe estar entre 0 y 1"
        self.epsilon = epsilon  # Probabilidad de exploración frente a explotación
        assert 0.0 < self.epsilon <= 1.0, "El valor epsilon debe estar entre 0 y 1"
        self.max_iteraciones = max_iteraciones # Número de iteraciones
        assert self.max_iteraciones > 0 , "Número de iteraciones tiene que ser positivo"
        self.max_pasos = max_pasos # Número de pasos que da el agente en cada iteración 
        assert self.max_pasos > 0 , "Número de pasos tiene que ser positivo"
        """    
            ################################################### ATRIBUTOS ######################################################################
                num_estados: 
                Type: int
                Description: Número de estados del problema.

                num_acciones: 
                Type: int
                Description: Número de acciones del problema.

                Q: 
                Type: Dict
                Description: Diccionario que almacena los valores de la función Q para cada par (estado, acción), valor inicial matriz a 0.
    """
        self.num_estados=transiciones[0].shape[0] if transiciones[0].ndim == 2 else len(transiciones)
        self.num_acciones = len(transiciones)

        # Inicialización de la matriz Q con ceros
        self.Q = np.zeros((self.num_estados, self.num_acciones))

    '''
    ################################################### MÉTODOS ######################################################################
    
        entrenar(destino,mapa): 
            Return: Dict
            Param: destino -> Tupla de dos coordenadas
                   mapa -> Lista de 0 y 1, donde 0 denota un espacio libre y 1 un obstáculo
            Entrena el algoritmo de Q-Learning y devuelve la función Q aprendida. Con un criterio de parada basado en el número de iteraciones y numero de pasos a realizar.

        es_terminal(estado, destino):
            Return: bool
            Param: estado -> Tupla de coordenadas
                   destino -> Tupla de dos coordenadas
            Método estático que comprueba si un estado es terminal si el estado del agente es igual al destino.

        seleccionar_estado(matriz):
            Return: int
            Param: matriz -> Lista de transiciones
            Selecciona el siguiente estado, en función de una matriz de probabilidades de transición.

        seleccionar_accion(estado) -> 
            Return: int
            Parama: estado -> Indice del estado
            Implementa una política epsilon-greedy para seleccionar la siguiente acción. A menor valor de epsilon equilibra la exploración (tratando acciones nuevas), y en caso contrario la explotación(optando por la mejor acción conocida). 

        obtener_politica() -> 
            Return: List
            Devuelve la política óptima aprendida a partir de la función Q.
        
    '''
    def entrenar(self, destino, mapa):
        # Recorre el bucle el número de iteraciones especificado
        for _ in range(self.max_iteraciones):
            # Selecciona un estado inicial aleatorio y elige una acción inicial para el estado actual
            estado_actual = random.randint(0, self.num_estados - 1)
            accion_actual = self.selecciona_accion(estado_actual)

            # Itera sobre el número máximo de pasos por iteración
            for _ in range(self.max_pasos):
                # Obtiene una tupla de coordenadas del indice del estado actual para verificar si el estado actual es terminal (llegó al destino)
                tupla_estado_actual = util.Problema.obtiene_estado_desde_indice(estado_actual, mapa)
                if self.es_terminal(tupla_estado_actual, destino):
                    break
                
                # Selecciona la matriz de transición según la acción y estado actual
                matriz_transicion = self.transiciones[accion_actual][estado_actual]
                # Elige el siguiente estado y acción
                
                estado_siguiente = self.selecciona_estado(matriz_transicion)
                accion_siguiente = self.selecciona_accion(estado_siguiente)
                
                # Obtiene la recompensa asociada al estado actual y acción actual
                recompensa = self.recompensas[estado_actual][accion_actual]

                # Calcula el valor objetivo Q-Learning
                max_q_siguiente = np.max(self.Q[estado_siguiente])
                valor_q = self.Q[estado_actual][accion_actual]
                valor_objetivo = recompensa + self.gamma * max_q_siguiente

                # Actualiza el valor Q para el estado actual y acción actual
                self.Q[estado_actual][accion_actual] += self.alpha * (valor_objetivo - valor_q)

                # Avanza al siguiente estado y acción
                estado_actual = estado_siguiente
                accion_actual = accion_siguiente

    # Determina si un estado dado es terminal, comparándolo con el destino.
    @staticmethod
    def es_terminal(estado, destino):
        return estado[0] == destino[0] and estado[1] == destino[1]
    
    # Selecciona un estado basado en una matriz de probabilidades (política).
    def selecciona_estado(self, matriz):
        return np.random.choice(range(self.num_estados), p=matriz) 

    # Selecciona una acción para un estado dado y usando la politica epsilon-greedy de caracter (explorativo(Aleatoriamente) o explotativo(Maximizando Recompensa)).
    def selecciona_accion(self, estado):
        return random.randint(0, self.num_acciones - 1) if random.uniform(0, 1) < self.epsilon else np.argmax(self.Q[estado])
    
    # Obtiene la política óptima aprendida por el agente.
    def obtener_politica(self):
        politica = np.argmax(self.Q, axis=1)
        acciones = ['esperar', 'N', 'NE', 'E', 'SE', 'S', 'SO', 'O', 'NO']
        return [acciones[a] for a in politica]