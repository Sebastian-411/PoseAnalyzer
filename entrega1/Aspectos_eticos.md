# **Aspectos éticos**

El desarrollo y uso de este sistema de inteligencia artificial implica una serie de consideraciones éticas que deben abordarse cuidadosamente para garantizar el respeto por los derechos, la privacidad y la integridad de las personas involucradas en el proceso de captura, anotación y análisis de datos.

1. **Consentimiento informado**
   Antes de grabar cualquier video, todos los participantes deben recibir una explicación clara, escrita y verbal sobre el propósito del proyecto, la naturaleza de los datos que se recopilarán, la forma en que se almacenarán y los posibles usos de los resultados. El consentimiento debe ser explícito, firmado y revocable en cualquier momento. Si un participante decide retirarse, sus datos deben ser eliminados permanentemente del conjunto de entrenamiento.

2. **Privacidad y anonimización**
   Los videos capturados no deben incluir información personal ni permitir la identificación facial directa de los sujetos. Se recomienda grabar a una distancia que capture el cuerpo completo sin necesidad de enfocar el rostro o, en caso de ser necesario, aplicar desenfoque facial previo a cualquier almacenamiento o procesamiento. Una vez extraídos los landmarks con *MediaPipe*, las coordenadas sustituyen la información visual original, eliminando toda posibilidad de identificación personal.

3. **Propósito limitado y uso responsable**
   Los datos deben ser utilizados exclusivamente con fines académicos o de investigación, sin ningún uso comercial ni de vigilancia. Cualquier intento de reutilizar los datos o el modelo para propósitos distintos —como control laboral, reconocimiento biométrico o seguridad— constituiría una violación directa a la ética del proyecto y requeriría un nuevo consentimiento y evaluación.

4. **Seguridad de la información**
   Todos los archivos de video, metadatos y resultados de entrenamiento deben almacenarse en repositorios privados protegidos con autenticación. No deben publicarse en repositorios abiertos como GitHub ni compartirse fuera del equipo autorizado. Se deben aplicar controles de acceso (solo lectura o edición) y registrar los cambios o descargas realizadas. Las copias locales deben encriptarse y eliminarse cuando el proyecto finalice.

5. **Minimización de datos**
   Solo se deben capturar y conservar los datos estrictamente necesarios para el objetivo del proyecto. Los videos originales deben eliminarse después de que las coordenadas y características numéricas sean extraídas y validadas. Esto reduce el riesgo de exposición de información sensible.

6. **No discriminación y equidad**
   El modelo debe ser evaluado con personas de distintas edades, complexiones, géneros y condiciones físicas para evitar sesgos. Un modelo entrenado con un grupo limitado podría fallar al generalizar en otros contextos, lo que sería injusto si se aplica en ámbitos clínicos o laborales. Se deben monitorear posibles sesgos de desempeño por tipo de cuerpo o género y reportarlos abiertamente.

7. **Transparencia y trazabilidad**
   El proceso de recolección, anotación, entrenamiento y validación debe ser completamente documentado. Cada versión del modelo debe acompañarse de información sobre los datos usados, el número de sujetos, los algoritmos aplicados y las métricas obtenidas. Esto permite auditar el proceso y detectar posibles errores o decisiones cuestionables.

8. **Derecho a la eliminación de datos**
   Cualquier participante podrá solicitar, en cualquier momento, la eliminación de sus videos o datos derivados. El sistema debe contemplar un protocolo rápido para localizar y eliminar toda referencia al participante en los conjuntos de entrenamiento, validación o prueba, así como en los respaldos.

9. **Transparencia del modelo y explicabilidad**
   Aunque el sistema clasifica actividades humanas mediante aprendizaje supervisado, es fundamental poder explicar cómo y por qué se toma cada decisión. Se deben incluir visualizaciones o reportes de las características más influyentes (por ejemplo, ángulos o velocidades) para evitar interpretaciones erróneas o falsas atribuciones de causa.

10. **Responsabilidad social y límites de aplicación**
    El sistema no debe utilizarse para vigilancia, control de productividad, evaluación automatizada de desempeño o cualquier otro uso que pueda vulnerar derechos fundamentales. Su propósito es educativo y de investigación en visión por computadora y análisis de movimiento humano.

11. **Conservación y eliminación programada**
    Los datos y modelos entrenados tendrán un ciclo de vida definido. Después de la evaluación final del proyecto, se realizará una revisión de qué archivos conservar y cuáles eliminar. Ningún dato sensible permanecerá almacenado más allá del tiempo justificado por razones académicas o legales.

12. **Supervisión ética continua**
    Se recomienda que todo el proceso esté bajo la revisión de un responsable ético o docente del curso, quien valide que se cumplan los principios de privacidad, consentimiento, equidad y seguridad durante todas las fases del proyecto.