<p align="center">
  <img src="Pawsitive%20Impact%20Banner.jpg" alt="Pawsitive Impact Banner" width="100%">
</p>

<h1 align="center">PAWSITIVE IMPACT</h1>
<p align="center">
  <strong>AI-powered spay & neuter hotspot predictor helping shelters prioritize targeted population control.</strong><br>
  <em>Built for the IBM AI Builders Challenge — AI for Good Challenge</em>
</p>

<p align="center">
  <a href="https://youtu.be/tRUzoypeV9k"><strong>Watch the 3-Minute Video Presentation</strong></a> •
  <a href="Anikka_Williams_AI_For_Good.pdf"><strong>View Presentation Slides (PDF)</strong></a>
</p>

---

## The Issue vs. Our Magic Solution

| The Issue | Our Magic Solution |
| :--- | :--- |
| Animals are reproducing out of control causing strain on municipal shelter budgets, overcrowding, and disrupting local urban ecosystems. Currently, non-profits and mobile spay/neuter units send surgical teams in response to spikes in intakes, causing higher operational costs and preventable euthanasia. | **Pawsitive Impact** uses machine learning to predict risk levels of overpopulation by zip code depending on shelter intake data. Developed using **IBM Bob**, it provides clear actions to take in the field based on risk levels—including targeted mobile clinic visits—and calculates direct shelter budget savings, helping shelters move from a reactive to a proactive approach. |

---

## How It Works

| 1. Predictive Risk Scoring | 2. Field Directives & ROI | 3. Custom Risk Simulator |
| :---: | :---: | :---: |
| Segments zip codes into **High**, **Medium**, and **Low** risk instantly. | Translates risk tiers into clear field actions like mobile surgical units and calculates direct shelter budget savings. | Directors input custom zip parameters to predict risk and overpopulation probability. |

---

## How IBM Bob Helped Build Pawsitive Impact
I used **IBM Bob** as an AI development partner to help create the architecture of the project, train and test my **Gradient Boosting Regressor** model, generate realistic baseline data on ZIP codes in Houston, and build the interactive **Streamlit** dashboard with custom risk filters and scenario simulations.

The prototype was built in **Python** using **Scikit-learn**, **Pandas**, **Plotly**, and **Streamlit** to create a local application that uses the predictive hotspots to give guidance in the field and save money for shelters without exposing sensitive data.

## Why This Is AI for Good
This project reflects my enthusiasm for using data science and AI to help with animal welfare issues. By helping shelters shift from reactive intake situations to being more proactive with community engagement, **Pawsitive Impact** uses AI to protect vulnerable animals, save money for municipalities, and strengthen local communities.

---

## Run the Prototype Locally

```bash
# 1. Clone the repository
git clone [https://github.com/anikkamw/pawsitive-impact-hotspot-predictor.git](https://github.com/anikkamw/pawsitive-impact-hotspot-predictor.git)
cd pawsitive-impact-hotspot-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Streamlit dashboard
streamlit run app.py
