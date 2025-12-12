Credit Scoring Business Understanding
Basel II Accord and the Need for Interpretability

The Basel II Accord emphasizes accurate and transparent measurement of credit risk using sound estimation systems and well-documented rating models. Because financial institutions must justify how risk scores are produced and ensure that credit decisions are explainable to regulators, an interpretable credit scoring model is essential. This requirement influences our project by prioritizing model transparency, documentation, and traceability over purely predictive performance. Every risk driver used in the model must be explainable and supported by clear business logic.

The Role of a Proxy Variable for Default

Our dataset does not include a direct "default" label—meaning we cannot directly observe whether a customer failed to repay their credit. To build a supervised model, we create a proxy variable that approximates default based on available behavioral signals (e.g., late payment counts, high past-due amounts, or long repayment delays).
While this allows us to proceed with model development, it introduces business risks:

The proxy may not perfectly represent true default behavior.

The model may learn patterns based on incomplete or biased assumptions.

Decisions based on an inaccurate proxy may misclassify customers, causing financial losses or denying credit to qualified borrowers.
Therefore, the proxy must be carefully defined, validated, and documented to manage these risks.

Model Trade-offs: Interpretability vs. Performance

In a regulated financial environment, choosing between simple and complex models involves important trade-offs:

Interpretable Models (e.g., Logistic Regression with Weight of Evidence):

Easily explainable to auditors, regulators, and management.

Highly transparent; each variable’s contribution is clear.

Easier to maintain, document, and justify.

May sacrifice predictive accuracy compared to modern ML techniques.

Complex Models (e.g., Gradient Boosting, XGBoost):

Typically achieve higher predictive performance and better capture nonlinear relationships.

More sensitive to data noise and require careful tuning.

Harder to interpret, document, and justify—posing regulatory and governance challenges.

Financial institutions often balance these trade-offs by using interpretable models for production scoring and complex models for internal risk analysis or challenger modeling.
