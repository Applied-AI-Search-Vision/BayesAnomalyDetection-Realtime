(function($) {
    // Initialize variables
    var $window = $(window),
        $body = $('body'),
        $wrapper = $('#wrapper'),
        $main = $('#main'),
        $articles = $main.children('article');

    // Breakpoints for responsive design
    const breakpoints = {
        xlarge:   [ '1281px',  '1680px' ],
        large:    [ '981px',   '1280px' ],
        medium:   [ '737px',   '980px'  ],
        small:    [ '481px',   '736px'  ],
        xsmall:   [ '361px',   '480px'  ],
        xxsmall:  [ null,      '360px'  ]
    };

    // Remove preload class after page loads
    $window.on('load', function() {
        window.setTimeout(function() {
            $body.removeClass('is-preload');
        }, 100);
    });

    // Article show/hide functionality
    const delay = 325;
    let locked = false;

    function showArticle(id, initial = false) {
        var $article = $articles.filter('#' + id);
        
        if ($article.length === 0) return;

        if (locked || initial) {
            $body.addClass('is-article-visible');
            $articles.removeClass('active');
            $article.show().addClass('active');
            locked = false;
            return;
        }

        locked = true;
        $body.addClass('is-article-visible');
        
        setTimeout(function() {
            $articles.hide().removeClass('active');
            $article.show().addClass('active');
            $window.scrollTop(0);
            
            setTimeout(function() {
                locked = false;
            }, delay);
        }, delay);
    }

    function hideArticle() {
        if (!$body.hasClass('is-article-visible')) return;

        locked = true;
        $articles.removeClass('active');
        
        setTimeout(function() {
            $articles.hide();
            $body.removeClass('is-article-visible');
            $window.scrollTop(0);
            
            setTimeout(function() {
                locked = false;
            }, delay);
        }, delay);
    }

    // Set up article close buttons
    $articles.each(function() {
        var $this = $(this);
        
        // Add close button
        $('<div class="close">Close</div>')
            .appendTo($this)
            .on('click', function() {
                location.hash = '';
            });

        // Prevent clicks inside article from bubbling
        $this.on('click', function(event) {
            event.stopPropagation();
        });
    });

    // Handle hash changes
    $window.on('hashchange', function(event) {
        let hash = location.hash;
        
        if (hash === '' || hash === '#') {
            event.preventDefault();
            hideArticle();
        } else if ($articles.filter(hash).length > 0) {
            event.preventDefault();
            showArticle(hash.substr(1));
        }
    });

    // Handle escape key
    $window.on('keyup', function(event) {
        if (event.keyCode === 27 && $body.hasClass('is-article-visible')) {
            location.hash = '';
        }
    });

    // Initialize
    if (location.hash && location.hash !== '#') {
        $window.on('load', function() {
            showArticle(location.hash.substr(1), true);
        });
    }

})(jQuery); 
