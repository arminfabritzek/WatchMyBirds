/*
# ------------------------------------------------------------------------------
# assets/custom_script for web_interface
# ------------------------------------------------------------------------------
*/
window.dash_clientside = window.dash_clientside || {};
window.dash_clientside.clientside = {
    scrollToPagination: function(triggerData) {
        if (triggerData) {
            const delayMilliseconds = 0;
            setTimeout(function() {
                const targetElement = document.getElementById('pagination-top');
                if (targetElement) {
                    // console.log(`Found pagination-top after ${delayMilliseconds}ms delay`);
                    targetElement.scrollIntoView({ behavior: 'auto', block: 'start' });
                } else {
                    console.error(`ERROR after ${delayMilliseconds}ms: Could not find element 'pagination-top'`);
                }
            }, delayMilliseconds);
        }
        return null;
    }
};

// --- Logic to make image click toggle the checkbox ---
document.addEventListener('DOMContentLoaded', () => {
    // console.log("Image click handler: DOMContentLoaded fired.");

    // Function to add listener (allows re-running)
    function initializeImageClickHandler() {
        const gridContainer = document.getElementById('edit-gallery-grid');

        if (gridContainer && !gridContainer.dataset.imageClickHandlerInitialized) {
            console.log("Image click handler: Found edit-gallery-grid, initializing.");
            gridContainer.dataset.imageClickHandlerInitialized = 'true'; // Mark as initialized

            gridContainer.addEventListener('click', (event) => {
                const clickedImage = event.target.closest('.thumbnail-image');
                const clickedTile = event.target.closest('.gallery-tile.edit-tile');

                // --- If an image inside an edit tile was clicked ---
                if (clickedImage && clickedTile) {
                    // --- Prevent click if it was directly on the checkbox area within ---
                    if (event.target.closest('.edit-checkbox')) {
                         // console.log("Image click handler: Click was inside checkbox area, ignoring for image click.");
                         return;
                    }

                    console.log("Image click handler: Click detected on image.");
                    event.preventDefault(); // Prevent default image actions

                    // Find the checkbox *input* within the same tile
                    const checkboxInput = clickedTile.querySelector('.edit-checkbox input[type="checkbox"]');

                    if (checkboxInput) {
                        // Programmatically click the checkbox input
                        // This triggers the browser's default toggle and event firing
                        console.log("Image click handler: Clicking associated checkbox input.");
                        checkboxInput.click();
                    } else {
                        console.error("Image click handler: Could not find checkbox input in tile:", clickedTile);
                    }
                }
            });
        } else if (!gridContainer) {
            // console.log("Image click handler: edit-gallery-grid not found yet.");
        }
    }

    // Run on initial load
    initializeImageClickHandler();

    // Re-run if grid might load dynamically
     const observerCheckInterval = setInterval(() => {
        const grid = document.getElementById('edit-gallery-grid');
        if (grid && !grid.dataset.imageClickHandlerInitialized) {
            console.log("Image click handler: Found grid via interval, running initializer.");
            initializeImageClickHandler();
        }
    }, 750);

    window.addEventListener('beforeunload', () => {
        clearInterval(observerCheckInterval);
    });

});
