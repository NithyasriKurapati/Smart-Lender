// Javascript for Smart Lender UI animations and client-side validation

document.addEventListener("DOMContentLoaded", () => {
    // 1. Gauge Animate (Result Page)
    const gaugeFill = document.querySelector(".gauge-fill");
    const gaugeValElement = document.getElementById("gauge-probability");
    
    if (gaugeFill && gaugeValElement) {
        const probability = parseFloat(gaugeValElement.getAttribute("data-probability"));
        
        // Circular gauge calculations
        // Circumference = 2 * pi * r = 2 * 3.14159 * 70 = 439.6
        const circumference = 439.6;
        const offset = circumference - (probability / 100) * circumference;
        
        // Animate gauge stroke offset after a brief delay for rendering
        setTimeout(() => {
            gaugeFill.style.strokeDashoffset = offset;
        }, 150);

        // Count-up animation for the probability text
        let count = 0;
        const duration = 1200; // ms
        const stepTime = Math.abs(Math.floor(duration / probability));
        
        const timer = setInterval(() => {
            count += 0.5;
            if (count >= probability) {
                gaugeValElement.textContent = probability + "%";
                clearInterval(timer);
            } else {
                gaugeValElement.textContent = count.toFixed(1) + "%";
            }
        }, Math.min(stepTime, 20));
    }

    // 2. Smooth scrolling to prediction form (Landing page)
    const predictBtn = document.getElementById("btn-predict-now");
    const formSection = document.getElementById("prediction-form-section");
    
    if (predictBtn && formSection) {
        predictBtn.addEventListener("click", (e) => {
            e.preventDefault();
            formSection.scrollIntoView({ behavior: "smooth" });
        });
    }

    // 3. Client-side Form Validation
    const loanForm = document.getElementById("loan-application-form");
    if (loanForm) {
        loanForm.addEventListener("submit", (event) => {
            let isValid = true;
            const errors = [];

            const appIncome = parseFloat(document.getElementById("ApplicantIncome").value);
            const coappIncome = parseFloat(document.getElementById("CoapplicantIncome").value);
            const loanAmt = parseFloat(document.getElementById("LoanAmount").value);
            const loanTerm = parseFloat(document.getElementById("Loan_Amount_Term").value);

            if (isNaN(appIncome) || appIncome < 0) {
                isValid = false;
                errors.push("Applicant Income must be a positive number.");
                document.getElementById("ApplicantIncome").classList.add("is-invalid");
            } else {
                document.getElementById("ApplicantIncome").classList.remove("is-invalid");
            }

            if (isNaN(coappIncome) || coappIncome < 0) {
                isValid = false;
                errors.push("Coapplicant Income must be a positive number (enter 0 if none).");
                document.getElementById("CoapplicantIncome").classList.add("is-invalid");
            } else {
                document.getElementById("CoapplicantIncome").classList.remove("is-invalid");
            }

            if (isNaN(loanAmt) || loanAmt <= 0) {
                isValid = false;
                errors.push("Loan Amount must be greater than zero.");
                document.getElementById("LoanAmount").classList.add("is-invalid");
            } else {
                document.getElementById("LoanAmount").classList.remove("is-invalid");
            }

            if (isNaN(loanTerm) || loanTerm <= 0) {
                isValid = false;
                errors.push("Loan Term must be a positive number of months.");
                document.getElementById("Loan_Amount_Term").classList.add("is-invalid");
            } else {
                document.getElementById("Loan_Amount_Term").classList.remove("is-invalid");
            }

            if (!isValid) {
                event.preventDefault();
                alert("Please correct the following errors:\n\n" + errors.join("\n"));
            } else {
                // Show loading spinner on button
                const submitBtn = loanForm.querySelector("button[type='submit']");
                if (submitBtn) {
                    submitBtn.disabled = true;
                    submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing Application...';
                }
            }
        });
    }
});
