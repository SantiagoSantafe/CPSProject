import cv2
from open_vocab_detector import OpenVocabularyDetector
    
if __name__ == "__main__":
    # 1. Instanciar el detector
    detector = OpenVocabularyDetector()
    
    # 2. Iniciar la Webcam (0 suele ser la cámara por defecto)
    print("Iniciando cámara... (Mira la ventana que se abre)")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        exit()

    print("--> PRESIONA 'ESPACIO' PARA TOMAR LA FOTO Y DETECTAR <---")
    print("--> PRESIONA 'Q' PARA SALIR <---")

    frame_to_process = None

    # Bucle para mostrar el video en vivo antes de capturar
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo recibir imagen de la cámara.")
            break

        # Mostrar instrucciones en la pantalla
        cv2.putText(frame, "Presiona ESPACIO para detectar", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow("Camara en vivo", frame)

        # Esperar tecla
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Tecla Espacio
            frame_to_process = frame.copy() # Guardamos la foto
            break
        elif key == ord('q'): # Tecla Q
            break
    
    # Soltamos la cámara (ya tenemos la foto)
    cap.release()
    cv2.destroyWindow("Camara en vivo")

    # 3. Si tomamos una foto, procesarla
    if frame_to_process is not None:
        print("Procesando imagen... Espera un momento...")
        
        # Objetos que SÍ quieres detectar
        target_queries = [
            "human face",
            "smartphone",
            "smartphone screen",
            "mobile phone",
            "prescription glasses on face",
            "human hand",
            "cable",
            "wire",
            "eyeglasses"
        ]

        # Objetos de fondo para "absorber" regiones irrelevantes
        background_queries = [
            "ceiling",
            "white wall", 
            "fluorescent light",
            "window",
            "empty background",
            "floor",
            "office ceiling tiles"
        ]

        queries = target_queries + background_queries
        
        # DETECTAR
        results = detector.detect_objects(frame_to_process, target_queries, background_queries)
        
        print(f"\nSe detectaron {len(results)} objetos.")
        
        # DIBUJAR
        for res in results:
            x, y, w, h = map(int, res['box'])
            label = res['label']
            score = res['score']

            basura = ["ceiling", "wall", "lights", "fluorescent lights", "background", "empty space"]
            if res['label'] in basura:
                continue
            # Rectángulo y Texto
            cv2.rectangle(frame_to_process, (x, y), (x+w, y+h), (0, 255, 0), 2)
            text = f"{label} ({score:.2f})"
            cv2.putText(frame_to_process, text, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Mostrar resultado final estático
        cv2.imshow("Resultado de Deteccion", frame_to_process)
        print("Presiona cualquier tecla para cerrar.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Saliste sin tomar foto.")