document.addEventListener("DOMContentLoaded", () => {
    const dateElement = document.getElementById("current-date");
    const today = new Date();
    dateElement.textContent = today.toLocaleDateString();

    const pdfContainer = document.getElementById("pdf-container");
    const pdfList = document.getElementById("pdf-list");
    const flashMessage = document.getElementById("flash-message");
    const addPdfButton = document.getElementById("add-pdf");
    const deleteSelectedButton = document.getElementById("delete-selected");
    const savePdfButton = document.getElementById("save-pdf");

    let pdfFiles = [];

    pdfContainer.addEventListener("dragover", (event) => {
        event.preventDefault();
        pdfContainer.style.backgroundColor = "#d3e9ff";
    });

    pdfContainer.addEventListener("dragleave", () => {
        pdfContainer.style.backgroundColor = "#f9f9f9";
    });

    pdfContainer.addEventListener("drop", (event) => {
        event.preventDefault();
        pdfContainer.style.backgroundColor = "#f9f9f9";
        const files = event.dataTransfer.files;
        handleFiles(files);
    });

    addPdfButton.addEventListener("click", () => {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = ".pdf";
        input.multiple = true;
        input.addEventListener("change", () => {
            handleFiles(input.files);
        });
        input.click();
    });

    deleteSelectedButton.addEventListener("click", () => {
        const checkboxes = document.querySelectorAll(".pdf-checkbox");
        checkboxes.forEach((checkbox, index) => {
            if (checkbox.checked) {
                pdfFiles.splice(index, 1);
                checkbox.parentElement.remove();
            }
        });
        showFlashMessage("Selected PDFs deleted successfully", "#dc3545");
    });

    savePdfButton.addEventListener("click", () => {
        if (pdfFiles.length > 0) {
            const formData = new FormData();
            pdfFiles.forEach((file, index) => {
                formData.append(`pdf${index}`, file);
            });

            fetch("/api/save-pdf", {
                method: "POST",
                body: formData
            })
            .then((response) => {
                if (!response.ok) throw new Error("Upload failed");
                return response.json();
            })
            .then((data) => {
                console.log("Upload success:", data);
                showFlashMessage("PDFs uploaded successfully", "#28a745");
            })
            .catch((error) => {
                console.error("Error uploading PDFs:", error);
                showFlashMessage("Error uploading PDFs", "#dc3545");
            });
        } else {
            showFlashMessage("No PDFs to save", "#ffc107");
        }
    });

    function handleFiles(files) {
        Array.from(files).forEach((file) => {
            if (file.type === "application/pdf") {
                pdfFiles.push(file);
                const li = document.createElement("li");
                li.innerHTML = `
                    <input type="checkbox" class="pdf-checkbox">
                    <span>${file.name}</span>
                `;
                pdfList.appendChild(li);
            }
        });
    }

    function showFlashMessage(message, bgColor) {
        flashMessage.textContent = message;
        flashMessage.style.backgroundColor = bgColor;
        flashMessage.style.display = "block";
        setTimeout(() => {
            flashMessage.style.display = "none";
        }, 3000);
    }
});
