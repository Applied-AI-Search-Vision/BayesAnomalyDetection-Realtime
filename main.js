// Smooth scroll animation for navigation
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
});

// Animate elements on scroll
const observerOptions = {
    threshold: 0.1,
    rootMargin: "0px 0px -50px 0px"
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('animate-in');
            observer.unobserve(entry.target);
        }
    });
}, observerOptions);

// Observe all sections and feature cards
document.querySelectorAll('.section, .feature-card').forEach(el => {
    observer.observe(el);
});

// Add hover effect to graphs
document.querySelectorAll('.visualization iframe').forEach(frame => {
    frame.addEventListener('mouseover', () => {
        frame.style.transform = `translateX(15%) scale(1.02)`;
    });
    
    frame.addEventListener('mouseout', () => {
        frame.style.transform = 'translateX(15%)';
    });
}); 