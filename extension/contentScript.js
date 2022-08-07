
const html = '<div id="sponsor-popup"><button id="skip-sponsor-btn">Skip Sponsor (30s)</button><button id="close-sponsor-popup-btn">X</button></div>'

var s = document.createElement('script');
s.src = chrome.runtime.getURL('playerScript.js');
s.onload = function () {
	this.remove();
};
(document.head || document.documentElement).appendChild(s);

let segments = null;
chrome.runtime.onMessage.addListener(
	function (request, sender, sendResponse) {
		if (request.from == 'popup') {
			if (request.subject == 'getSegments') {
				sendResponse({ segments, title: document.getElementsByClassName('title style-scope ytd-video-primary-info-renderer')[0].innerText })
			}
			else if (request.subject == 'seekTo') {
				const clickEvent = new CustomEvent('seekTo', { detail: request.to })
				document.documentElement.dispatchEvent(clickEvent)
			}

			return
		}
		if (location.href.endsWith(request.video_id)) {
			segments = request.segments
			if (segments == null) {
				if (sponsorPopupElement) {
					sponsorPopupElement.style['display'] = 'none'
				}
			}
		}

	}
);

let readyInterval = setInterval(initSkipElement, 500)
let sponsorPopupElement = null
let skipBtnElement = null
let closeBtnElement = null
let currentTime = 0

function initSkipElement() {
	try {
		document.getElementById('ytd-player').parentElement.insertAdjacentHTML('afterbegin', html)
		sponsorPopupElement = document.getElementById('sponsor-popup')
		skipBtnElement = document.getElementById('skip-sponsor-btn')
		closeBtnElement = document.getElementById('close-sponsor-popup-btn')
		clearInterval(readyInterval)
	} catch (error) { }
}

const closedSegments = new Set()

document.documentElement.addEventListener('currentVideoTime', (e) => {
	if (sponsorPopupElement == null || segments == null) {
		return
	}
	currentTime = e.detail
	var inSponsor = false
	segments.forEach(([start, end], index) => {
		if (closedSegments.has(index)) return
		if (currentTime >= start && currentTime <= end) {
			sponsorPopupElement.style['display'] = 'unset'
			const clickEvent = new CustomEvent('seekTo', { detail: end })

			skipBtnElement.onclick = () => {
				document.documentElement.dispatchEvent(clickEvent)
				sponsorPopupElement.style['display'] = 'none'
			}

			closeBtnElement.onclick = () => {
				closedSegments.add(index)
				sponsorPopupElement.style['display'] = 'none'
			}

			skipBtnElement.innerText = 'Skip Sponsor (' + Math.ceil(end - currentTime) + 's)'
			inSponsor = true
		}
	});

	if (!inSponsor) {
		sponsorPopupElement.style['display'] = 'none'
	}
})



