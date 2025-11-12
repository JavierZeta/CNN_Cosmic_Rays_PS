import random
import os
import shutil

#Creación de la nueva carpeta donde irá el dataset
dataset_entrenamiento=r"D:\FIE\Photsat\dataset_entrenamiento"

os.makedirs(dataset_entrenamiento, exist_ok=True)

N=228

dataset_antiguo=r"D:\FIE\Photsat\sumadas"

dataset_nuevo=r"D:\FIE\Photsat\sumadas_nuevo"

lista_dataset_antiguo=os.listdir(dataset_antiguo)

lista_dataset_nuevo=os.listdir(dataset_nuevo)

watchdog_antiguo=202

watchdog_nuevo=82

iteraciones=0

flag_antiguo=0

flag_nuevo=0

for i in range(1,N+1):
    
    seed=random.randint(0, 1)
    
    if seed==0:
        
        while True:
            
            if flag_antiguo==1:
                
                print("Ya se han copiado todos los archivos")
                
                i-=1
                
                break
            
            iteraciones+=1
        
            archivo_seleccionado=random.choice(lista_dataset_antiguo)
        
            ruta_completa=os.path.join(dataset_antiguo,archivo_seleccionado)
            
            ruta_dataset_entrenamiento=os.path.join(dataset_entrenamiento,archivo_seleccionado)
            
            if not os.path.exists(ruta_dataset_entrenamiento):
                
                shutil.copy(ruta_completa, ruta_dataset_entrenamiento)
                
                iteraciones=0
                
                print(i)
                
                break
            
            if iteraciones>=watchdog_antiguo:
                
                flag_antiguo=1
                
                print("Ya no quedan archivos por copiar")
                
                break
    
    if seed==1:
        
        while True:
            
            if flag_nuevo==1:
                
                print("Ya se han copiado todos los archivos")
                
                i-=1
                
                break
            
            iteraciones+=1
        
            archivo_seleccionado=random.choice(lista_dataset_nuevo)
        
            ruta_completa=os.path.join(dataset_nuevo,archivo_seleccionado)
        
            ruta_dataset_entrenamiento=os.path.join(dataset_entrenamiento,archivo_seleccionado)
        
            if not os.path.exists(ruta_dataset_entrenamiento):
                
                shutil.copy(ruta_completa, ruta_dataset_entrenamiento)
                
                iteraciones=0
                
                print(i)
            
                break
            
            if iteraciones>=watchdog_nuevo:
                
                flag_nuevo=1
                
                print("Ya no quedan archivos por copiar")
                
                break