import numpy as np
import random
from utils import problema as util

'''
    Clase que implementa el algoritmo de Monte Carlo para la resolución de problemas de aprendizaje por refuerzo.
    Permite el entrenamiento utilizando la política de primera visita y cada visita.
'''

class MonteCarlo:

    def __init__(self, transiciones, recompensas, gamma=0.95, epsilon=0.1, max_iteraciones=1000, max_pasos=100):
        '''
        ########################################## PARÁMETROS ##########################################
        transiciones: 
                Type: List
                Description: Matriz de probabilidades de transición. Cada fila representa un estado y cada columna una acción, y su valor la probabilidad de transitar al siguiente estado.

            recompensas: 
                Type: List
                Description: Matriz de recompensas. Cada fila representa un estado y cada columna una acción y su valor la recompensa asociada a ambas.

            politica_in: 
                Type: Dict
                Description: Política inicial. DefaultValue=None

            gamma: 
                Type: float
                Description: Factor de descuento. DefaultValue=0.95

            epsilon: 
                Type: float
                Description: Probabilidad de exploración. DefaultValue=0.1

            max_iteraciones: 
                Type: int
                Description: Número máximo de iteraciones. DefaultValue=1000
        '''
        self.transiciones = transiciones
        self.recompensas = recompensas
        self.gamma = gamma  # Factor de descuento
        self.epsilon = epsilon  # Probabilidad de exploración
        self.max_iteraciones = max_iteraciones
        self.max_pasos = max_pasos

        self.num_acciones = len(transiciones)
        self.num_estados = transiciones.shape[1] if transiciones.ndim == 3 else transiciones[0].shape[0]

        # 1º Inicio arbitrario de la política inicial
        self.politica = {p: np.random.choice(self.num_acciones) for p in range(self.num_estados)}

        # 2º Inicialización del diccionario de recompensas acumuladas ciomo una lista vacía
        self.recompensas_acumuladas = np.zeros((self.num_estados, self.num_acciones), dtype=object)
        for e in range(self.num_estados):
            for a in range(self.num_acciones):
                self.recompensas_acumuladas[e][a] = []
                
        # 3º Inicialización del diccionario de valores Q, 
        self.valores_q = np.zeros((self.num_estados, self.num_acciones))
                
                
        '''
        ########################################## MÉTODOS ##########################################
        
        entrenar_primera_visita(destino, mapa): 
            Entrena utilizando el método de primera visita. 

        entrenar_cada_visita(destino, mapa): 
            Entrena utilizando el método de cada visita. 

        siguiente_estado(estado, accion): 
            Return: int
            Param: estado -> Estado actual.
                   accion -> Acción tomada.
            Selecciona el siguiente estado en función de la acción y el estado actual usando una política epsilon-greedy.

        generar_episodio(destino, mapa): 
            Return: List
            Param: destino -> Tupla de coordenadas.
                   mapa -> Lista de 0 y 1, donde 0 denota un espacio libre y 1 un obstáculo.
            Genera un episodio (lista de tuplas estado, acción, recompensa).

        es_estado_terminal(estado, destino): 
            Return: bool
            Param: estado -> Tupla de coordenadas.
                   destino -> Tupla de coordenadas.
            Comprueba si el estado es terminal, es decir, si coincide con el destino.

        siguiente_accion(estado): 
            Return: int
            Param: estado -> Estado actual.
            Selecciona la siguiente acción usando una política epsilon-greedy.

        obtener_politica(): 
            Return: List
            Devuelve la política óptima aprendida a partir de los valores Q.
        '''

    def entrenar_primera_visita(self,destino,mapa): #Primera visita
        for _ in range(self.max_iteraciones):
            episodio = self.generar_episodio(destino,mapa) # Pasa la lista (estado,accion,recompensa)
            visitados = set()
            for t in range(len(episodio)):
                estado, accion, recompensa = episodio[t]
                if (estado, accion) not in visitados: # 7º Comprobamos que el estado y la accion es la primera vez que ocurre en la secuencia
                    visitados.add((estado, accion)) # Añadimos para saber que ya lo hemos utilizado
                    U = sum([self.gamma**(i-t)*recompensa for i in range(t, len(episodio))]) # 9º sumatorio(gamma^(i-t)*Recompensa)
                    # U = sum(self.gamma ** (i - t) * r for i, (_, _, r) in enumerate(episodio[t:], start=t))
                    self.recompensas_acumuladas[estado][accion].append(U) # 10º añadimos a recompensas acumuladas la que acabamos de calcular
                    self.valores_q[estado][accion] = np.mean(self.recompensas_acumuladas[estado][accion]) # 11º Calculamos la media de las recompensas para añadirlo a los valores de Q
                    self.politica[estado] = np.argmax(self.valores_q[estado]) # 12º  Actualizamos la política con el valor máximo de los valores de Q

    def entrenar_cada_visita(self,destino,mapa): #Cada visita
        for _ in range(self.max_iteraciones):
            episodio = self.generar_episodio(destino,mapa) # Pasa la lista (estado,accion,recompensa)
            for t in range(len(episodio)):
                estado, accion, recompensa = episodio[t]
                U = sum([self.gamma**(i-t)*recompensa for i in range(t, len(episodio))]) # 9º sumatorio(gamma^(i-t)*Recompensa)
                # U = sum(self.gamma ** (i - t) * r for i, (_, _, r) in enumerate(episodio[t:], start=t))
                self.recompensas_acumuladas[estado][accion].append(U) # 10º añadimos a recompensas acumuladas la que acabamos de calcular
                self.valores_q[estado][accion] = np.mean(self.recompensas_acumuladas[estado][accion]) # 11º Calculamos la media de las recompensas para añadirlo a los valores de Q
                self.politica[estado] = np.argmax(self.valores_q[estado]) # 12º  Actualizamos la política con el valor máximo de los valores de Q
            
                        
    def siguiente_estado(self, estado, accion):
        mtrx = self.transiciones[accion][estado]
        return np.random.choice(range(len(mtrx)), p=mtrx)  if random.uniform(0, 1) < self.epsilon else np.argmax(mtrx)
                
    def generar_episodio(self,destino,mapa):
        episodio = []
        estado_actual =random.randint(0, self.num_estados - 1) # Elegir un estado aleatorio
        tupla_estado_actual=util.Problema.obtiene_estado_desde_indice(estado_actual,mapa)
        while(self.es_estado_terminal(tupla_estado_actual,destino)): # Comprobar que el estado no es terminal (es el destino), en caso de que lo sea:
            estado_actual =random.randint(0, self.num_estados - 1) # Elegir un estado aleatorio
            tupla_estado_actual=util.Problema.obtiene_estado_desde_indice(estado_actual,mapa)
            
        accion_actual = np.random.choice(range(self.num_acciones)) #Elegir accion aleatoria
        recompensa = self.recompensas[estado_actual][accion_actual] # optiene la recompensa del estado y accion escogido en cada iteración

        for _ in range(self.max_pasos): # Condicion de parada
            if self.es_estado_terminal(tupla_estado_actual,destino):
                break
            episodio.append((estado_actual, accion_actual, recompensa)) # 6º genera el episodio (estado,accion,reconmpensa)
            estado_actual = self.siguiente_estado(estado_actual, accion_actual) 
            accion_actual = self.siguiente_accion(estado_actual)
            recompensa = self.recompensas[estado_actual][accion_actual]


        return episodio # Lista de (estado,accion,recompensa)


    @staticmethod
    def es_estado_terminal(estado, destino):
        return estado[0] == destino[0] and estado[1] == destino[1]
    
    def siguiente_accion(self, estado):
        return random.randint(0, self.num_acciones - 1) if random.uniform(0, 1) < self.epsilon else np.argmax(self.valores_q[estado])

    def obtener_politica(self):
        acciones = ['esperar', 'N', 'NE', 'E', 'SE', 'S', 'SO', 'O', 'NO']
        return [acciones[a] for a in self.politica.values()]
    
