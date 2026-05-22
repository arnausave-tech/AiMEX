import tensorflow as tf
import numpy as np
import random
import sympy as sp

# 🧠 INTENCIONES
intents = {
    "saludo": {
        "patterns": ["hola", "hey", "buenas"],
        "responses": ["Hola 👋", "Qué tal!"]
    },
    "calculo": {
        "patterns": ["calcula", "resuelve", "cuanto es", "evalua"],
        "responses": ["OK, calculando..."]
    }
}

words = sorted(set("hola hey buenas calcula resuelve cuanto es evalua".split()))
labels = ["saludo", "calculo"]

def bag_of_words(sentence):
    bag = [0]*len(words)
    s_words = sentence.lower().split()
    for i, w in enumerate(words):
        if w in s_words:
            bag[i] = 1
    return np.array(bag)

# 🧠 MODELO SIMPLE
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, input_shape=(len(words),), activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(len(labels), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# dataset pequeño
X = []
Y = []

for i, label in enumerate(labels):
    for pattern in intents[label]["patterns"]:
        X.append(bag_of_words(pattern))
        Y.append(i)

X = np.array(X)
Y = np.array(Y)

model.fit(X, Y, epochs=300, verbose=0)

print("🤖 Bot matemático listo!")

# 🧮 FUNCIÓN MATEMÁTICA REAL
def resolver_matematica(texto):
    try:
        expr = texto.lower()

        # limpiar palabras comunes
        for w in ["calcula", "resuelve", "cuanto es", "evalua"]:
            expr = expr.replace(w, "")

        expr = expr.strip()

        # SymPy evalúa matemáticas avanzadas
        resultado = sp.sympify(expr).evalf()
        return resultado
    except:
        return "No pude resolverlo 😕"

# 💬 CHAT
while True:
    msg = input("Tú: ")

    if msg.lower() == "salir":
        break

    bow = bag_of_words(msg)
    pred = model.predict(np.array([bow]), verbose=0)
    tag = labels[np.argmax(pred)]

    if tag == "calculo":
        print("Bot:", resolver_matematica(msg))
    else:
        print("Bot:", random.choice(intents[tag]["responses"]))
