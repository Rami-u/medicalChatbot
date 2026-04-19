"""
responses.py — Predefined smart responses for each intent
-----------------------------------------------------------
Add or edit entries here to change what the bot says for
each classified category.  Each key is an intent name that
must match the labels in intents.json.
"""

import random

RESPONSES: dict[str, list[str]] = {
    "allergy": [
        "🤧 Your symptoms suggest an **allergic reaction**. Common triggers include pollen, dust, pet dander, or certain foods. Try to identify and avoid the allergen. Antihistamines can provide quick relief — but please consult a doctor if symptoms persist or worsen.",
        "🌿 It looks like you may be experiencing **allergy symptoms** such as itching, rash, or red eyes. Over-the-counter antihistamines can help, but I strongly recommend seeing a healthcare provider to identify the specific allergen and get a tailored treatment plan.",
        "⚠️ Based on what you've described, this could be an **allergic response**. Keep track of what you were exposed to recently (food, environment, products). Avoid known allergens and consider consulting an allergist for proper testing.",
    ],
    "anemia": [
        "🩸 Your symptoms — such as fatigue, dizziness, or pale skin — may indicate **anemia** (low red blood cell count). Eating iron-rich foods like spinach, lentils, and red meat can help. However, please see a doctor for a blood test to confirm and get appropriate treatment.",
        "💊 Fatigue and dizziness are classic signs of **anemia**. This is often caused by iron, B12, or folate deficiency. A simple blood test can confirm it. In the meantime, ensure a balanced diet rich in iron and vitamin C. Medical evaluation is recommended.",
        "🥗 Pale skin paired with dizziness may point to **anemia**. Don't ignore these signs — schedule a blood panel with your doctor. Iron supplements and dietary changes can be very effective when the right cause is identified.",
    ],
    "arthritis": [
        "🦴 Joint pain, swelling, and stiffness are hallmark symptoms of **arthritis**. There are many types (osteoarthritis, rheumatoid, etc.). Gentle exercise, warm compresses, and anti-inflammatory medications can help manage pain. A rheumatologist can provide a definitive diagnosis.",
        "🏃 Your symptoms could be related to **arthritis**. Low-impact exercises like swimming or yoga, along with maintaining a healthy weight, can significantly reduce joint stress. Please consult an orthopedic specialist or rheumatologist for proper diagnosis and a treatment plan.",
        "💊 Stiffness and swelling in the joints often point to **arthritis**. Early treatment is key to preventing joint damage. Anti-inflammatory medications and physical therapy are common treatments. See your doctor soon for imaging and blood tests.",
    ],
    "cold": [
        "🤒 Your symptoms — runny nose, sore throat, and cough — are classic signs of the **common cold**, caused by a viral infection. Rest, drink plenty of fluids, and consider over-the-counter cold medications. Most colds resolve in 7–10 days.",
        "🍵 It sounds like you might have a **cold**. Stay hydrated, get plenty of rest, and try warm liquids like honey-lemon tea to soothe your throat. Zinc lozenges may shorten duration if taken early. See a doctor if symptoms worsen after a week.",
        "💊 A runny nose combined with sneezing and a sore throat usually indicates a **viral cold**. Antibiotics won't help (it's a virus!) — but rest, hydration, and symptom relievers will. Contact a doctor if you develop a high fever or difficulty breathing.",
    ],
    "diabetes": [
        "🍬 Symptoms like frequent urination, excessive thirst, and fatigue may indicate **diabetes**. This is a serious condition that requires medical attention. Please see a doctor immediately for blood sugar testing. Early detection makes a huge difference in management.",
        "📊 Your symptoms could be associated with **high blood sugar (diabetes)**. Lifestyle changes like a low-sugar diet, regular exercise, and proper hydration are important. However, you **must** consult a healthcare provider for proper diagnosis and medication management.",
        "⚠️ Excessive thirst and frequent urination are warning signs of **diabetes mellitus**. A fasting blood glucose test is the first step. With proper management — diet, medication, and monitoring — people with diabetes can live healthy lives. See your doctor soon.",
    ],
    "flu": [
        "🤧 Your symptoms — fever, body aches, and fatigue — strongly suggest **influenza (flu)**. Rest is crucial. Stay well-hydrated and use fever reducers like acetaminophen. If you're in a high-risk group, antiviral medications (like Tamiflu) are most effective in the first 48 hours.",
        "🌡️ Flu symptoms like sudden fever, chills, and severe fatigue can knock you out for days. Rest at home, drink lots of fluids, and avoid contact with others. See a doctor promptly if you have difficulty breathing or persistent chest pain.",
        "💊 The **flu** is more severe than a common cold. Key signs include high fever, muscle aches, and exhaustion. Annual flu vaccination is the best prevention. For treatment, rest and hydration are essential — and see a doctor if you're at risk for complications.",
    ],
    "hypertension": [
        "❤️ Symptoms like headaches, dizziness, or blurred vision can be associated with **high blood pressure (hypertension)**. This is a 'silent' condition — often there are no symptoms at all. Regular monitoring is essential. Reduce salt, stress, and alcohol; increase exercise.",
        "📉 **Hypertension** is a serious cardiovascular risk factor. If you're experiencing related symptoms, check your blood pressure immediately. Lifestyle changes (DASH diet, exercise, reduced sodium) combined with medication (if prescribed) can effectively control it.",
        "⚠️ High blood pressure rarely causes obvious symptoms until it's severe. If you suspect hypertension, see a doctor for a proper reading and assessment. Left unmanaged, it can lead to stroke, heart attack, or kidney damage. Early action is critical.",
    ],
    "migraine": [
        "🧠 A **migraine** is more than just a headache — it's a neurological event. Triggers include stress, hormonal changes, certain foods, and bright lights. In a quiet, dark room with a cold compress can help. Pain relievers or triptans are commonly used treatments.",
        "💊 Severe, throbbing head pain with nausea and light sensitivity is a classic **migraine**. Keep a migraine diary to identify triggers. Over-the-counter pain relievers may help mild cases, but a neurologist can prescribe more targeted medications for frequent attacks.",
        "🌙 Migraines can be debilitating. Rest in a dark, quiet room and apply a cold cloth to your forehead. Stay hydrated. If migraines are frequent (more than 4 per month), please consult a neurologist — preventive medications can significantly reduce their frequency.",
    ],
    "pneumonia": [
        "🫁 Chest pain, difficulty breathing, and fever may indicate **pneumonia** — a serious lung infection. This requires prompt medical evaluation. Do not delay seeking care, especially if you're elderly, very young, or have a compromised immune system.",
        "🏥 **Pneumonia** is an infection that inflames the air sacs in one or both lungs. It requires medical treatment, usually antibiotics (for bacterial pneumonia) or antiviral drugs. Rest, hydration, and fever management are important. Please see a doctor as soon as possible.",
        "⚠️ If you're experiencing persistent cough with phlegm, fever, and shortness of breath, **pneumonia** should be ruled out urgently. A chest X-ray is the standard diagnostic tool. Severe cases may require hospitalization. Please seek medical help immediately.",
    ],
}

# Fallback for unknown intents
FALLBACK_RESPONSES = [
    "🤔 I'm not quite sure what condition your symptoms point to. I recommend describing your symptoms in more detail, or consulting a qualified healthcare professional for an accurate diagnosis.",
    "💬 Based on your input, I couldn't confidently classify your symptoms. Please provide more specific details about what you're experiencing, or visit a doctor for proper evaluation.",
    "🩺 Your symptoms don't clearly match a single condition in my database. A medical professional can give you a thorough examination and accurate diagnosis. Please don't rely solely on this chatbot for medical decisions.",
]


def get_response(intent: str) -> str:
    """Return a (randomly selected) response for the given intent."""
    responses = RESPONSES.get(intent, FALLBACK_RESPONSES)
    return random.choice(responses)
