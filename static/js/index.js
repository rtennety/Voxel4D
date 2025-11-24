$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");
    });

    var options = {
			slidesToScroll: 1,
			slidesToShow: 3,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }
})

// ========== PROTECTION CODE ==========

// Disable right-click context menu
document.addEventListener('contextmenu', function(e) {
    e.preventDefault();
    return false;
}, false);

// Disable text selection
document.addEventListener('selectstart', function(e) {
    e.preventDefault();
    return false;
}, false);

// Disable drag
document.addEventListener('dragstart', function(e) {
    e.preventDefault();
    return false;
}, false);

// Block keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Block F12 (DevTools)
    if (e.key === 'F12') {
        e.preventDefault();
        showBlackOverlay();
        return false;
    }
    
    // Block Ctrl+Shift+I (DevTools)
    if (e.ctrlKey && e.shiftKey && e.key === 'I') {
        e.preventDefault();
        showBlackOverlay();
        return false;
    }
    
    // Block Ctrl+Shift+J (Console)
    if (e.ctrlKey && e.shiftKey && e.key === 'J') {
        e.preventDefault();
        showBlackOverlay();
        return false;
    }
    
    // Block Ctrl+Shift+C (Inspect Element)
    if (e.ctrlKey && e.shiftKey && e.key === 'C') {
        e.preventDefault();
        showBlackOverlay();
        return false;
    }
    
    // Block Ctrl+U (View Source)
    if (e.ctrlKey && e.key === 'u') {
        e.preventDefault();
        return false;
    }
    
    // Block Ctrl+S (Save)
    if (e.ctrlKey && e.key === 's') {
        e.preventDefault();
        return false;
    }
    
    // Block Ctrl+P (Print)
    if (e.ctrlKey && e.key === 'p') {
        e.preventDefault();
        return false;
    }
    
    // Block Ctrl+C (Copy)
    if (e.ctrlKey && e.key === 'c') {
        e.preventDefault();
        return false;
    }
    
    // Block Ctrl+A (Select All)
    if (e.ctrlKey && e.key === 'a') {
        e.preventDefault();
        return false;
    }
    
    // Block Print Screen
    if (e.key === 'PrintScreen' || (e.ctrlKey && e.key === 'PrintScreen')) {
        e.preventDefault();
        showBlackOverlay();
        return false;
    }
    
    // Block Windows+Shift+S (Snipping Tool)
    if (e.shiftKey && e.key === 'S' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        showBlackOverlay();
        return false;
    }
}, false);

// Show black overlay
function showBlackOverlay() {
    var overlay = document.getElementById('screenshot-protection-overlay');
    if (overlay) {
        overlay.classList.add('active');
        // Hide after a short delay when focus returns
        setTimeout(function() {
            if (document.hasFocus()) {
                overlay.classList.remove('active');
            }
        }, 500);
    }
}

// Detect DevTools opening
(function() {
    var devtools = {open: false, orientation: null};
    var threshold = 160;
    setInterval(function() {
        if (window.outerHeight - window.innerHeight > threshold || 
            window.outerWidth - window.innerWidth > threshold) {
            if (!devtools.open) {
                devtools.open = true;
                showBlackOverlay();
            }
        } else {
            if (devtools.open) {
                devtools.open = false;
                var overlay = document.getElementById('screenshot-protection-overlay');
                if (overlay) {
                    overlay.classList.remove('active');
                }
            }
        }
    }, 500);
})();

// Detect page visibility changes (tab switching, minimizing, etc.)
document.addEventListener('visibilitychange', function() {
    if (document.hidden) {
        showBlackOverlay();
    } else {
        var overlay = document.getElementById('screenshot-protection-overlay');
        if (overlay) {
            setTimeout(function() {
                overlay.classList.remove('active');
            }, 300);
        }
    }
}, false);

// Detect window blur (losing focus)
window.addEventListener('blur', function() {
    showBlackOverlay();
}, false);

// Detect window focus (regaining focus)
window.addEventListener('focus', function() {
    setTimeout(function() {
        var overlay = document.getElementById('screenshot-protection-overlay');
        if (overlay) {
            overlay.classList.remove('active');
        }
    }, 300);
}, false);

// Watermark overlay (appears intermittently)
setInterval(function() {
    var watermark = document.getElementById('watermark-overlay');
    if (watermark && Math.random() > 0.95) { // 5% chance every interval
        watermark.classList.add('active');
        setTimeout(function() {
            watermark.classList.remove('active');
        }, 100);
    }
}, 1000);

// Prevent screenshot via console
(function() {
    var noop = function() {};
    Object.defineProperty(window, 'console', {
        value: {
            log: noop,
            warn: noop,
            error: noop,
            info: noop,
            debug: noop,
            assert: noop,
            clear: noop,
            count: noop,
            dir: noop,
            dirxml: noop,
            group: noop,
            groupCollapsed: noop,
            groupEnd: noop,
            profile: noop,
            profileEnd: noop,
            time: noop,
            timeEnd: noop,
            timeStamp: noop,
            trace: noop
        }
    });
})();
