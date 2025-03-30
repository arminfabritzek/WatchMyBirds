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