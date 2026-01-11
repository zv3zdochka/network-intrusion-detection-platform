/**
 * Main JavaScript utilities
 */

// Format numbers with locale
function formatNumber(num) {
    return num.toLocaleString();
}

// Format bytes
function formatBytes(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// Format duration
function formatDuration(seconds) {
    if (seconds < 60) return seconds + 's';
    if (seconds < 3600) return Math.floor(seconds / 60) + 'm ' + (seconds % 60) + 's';
    return Math.floor(seconds / 3600) + 'h ' + Math.floor((seconds % 3600) / 60) + 'm';
}

// Protocol name mapping
function protocolName(proto) {
    const protocols = {
        1: 'ICMP',
        6: 'TCP',
        17: 'UDP'
    };
    return protocols[proto] || proto;
}

// Show notification
function showNotification(message, type = 'info') {
    // Simple alert for now, can be replaced with toast
    if (type === 'error') {
        alert('Error: ' + message);
    }
}

// Debounce function
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}