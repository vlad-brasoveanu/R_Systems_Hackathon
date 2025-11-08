Proiect: Clasificator de Imagini (Robot vs. Om)

Acesta este un prototip full-stack al unei aplicații de machine learning care poate diferenția imagini cu roboți de cele cu oameni.

📝 Descriere Generală

Proiectul este format dintr-un model de deep learning antrenat să recunoască diferențele vizuale dintre oameni și roboți. Acest model este apoi integrat într-o aplicație web simplă, unde un utilizator poate încărca o imagine și primi o predicție în timp real.

Toate predicțiile sunt înregistrate într-o bază de date cloud (Supabase) pentru analize viitoare.

🚀 Tehnologii Folosite

Model Machine Learning: PyTorch (folosind un model pre-antrenat ResNet-18 pentru Transfer Learning).

Backend & API: Flask (Python) - servește modelul și interfața web.

Frontend: HTML, CSS, JavaScript (pentru încărcarea imaginilor și afișarea rezultatelor).

Bază de Date: Supabase (PostgreSQL) - accesată printr-un API REST.

Expunere Publică (Demo): ngrok - creează un tunel securizat către serverul local Flask.

🏃‍♂️ Cum Funcționează (Fluxul de Rulare)

Antrenarea (Local):

Scriptul train.py antrenează modelul ResNet-18 pe un set de date personalizat (din folderul dataset/).

Salvează modelul antrenat ca robot_human_classifier.pth.

Generează un raport de performanță (training_plot.png și training_report.json).

Aplicația Live (Server):

Scriptul app.py pornește un server Flask care:

Încarcă modelul robot_human_classifier.pth.

Servește interfața web (index.html).

Oferă un endpoint API (/predict).

Interacțiunea Utilizatorului (Web):

Utilizatorul accesează serverul (printr-un link ngrok).

Încarcă o imagine în index.html.

JavaScript-ul trimite imaginea la endpoint-ul /predict.

Serverul Flask primește imaginea, o procesează cu modelul, și trimite înapoi predicția (ex: "Robot 98%").

Serverul salvează simultan rezultatul în baza de date Supabase.
