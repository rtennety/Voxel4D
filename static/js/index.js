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
    
    // ========== IMAGE PROTECTION ==========
    
    // Wrap all images in protective containers
    $('img').each(function() {
        var $img = $(this);
        var $wrapper = $('<div class="image-protected"></div>');
        var $overlay = $('<div class="image-overlay"></div>');
        
        $img.wrap($wrapper);
        $img.after($overlay);
    });
    
    // Disable right-click on images and overlays
    $(document).on('contextmenu', 'img, .image-overlay, .image-protected', function(e) {
        e.preventDefault();
        e.stopPropagation();
        return false;
    });
    
    // Prevent drag on images
    $(document).on('dragstart', 'img', function(e) {
        e.preventDefault();
        return false;
    });
    
    // Prevent image selection
    $(document).on('selectstart', 'img', function(e) {
        e.preventDefault();
        return false;
    });
    
    // Block common image saving keyboard shortcuts
    $(document).on('keydown', function(e) {
        // Block Ctrl+S (Save)
        if (e.ctrlKey && e.key === 's') {
            e.preventDefault();
            return false;
        }
        // Block Ctrl+Shift+I (Inspect - might reveal image URLs)
        if (e.ctrlKey && e.shiftKey && e.key === 'I') {
            e.preventDefault();
            return false;
        }
    });
    
    // Prevent opening image in new tab
    $(document).on('click', 'img', function(e) {
        e.preventDefault();
        return false;
    });
    
    // Block image loading via right-click menu (additional protection)
    document.addEventListener('DOMContentLoaded', function() {
        var images = document.querySelectorAll('img');
        images.forEach(function(img) {
            // Remove ability to open image in new tab
            img.setAttribute('oncontextmenu', 'return false;');
            img.setAttribute('ondragstart', 'return false;');
            img.setAttribute('onselectstart', 'return false;');
        });
    });
})
