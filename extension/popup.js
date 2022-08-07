function setContent(message) {
	document.getElementById('content').innerHTML = message
}

function secondsToMMSS(seconds) {
	return new Date(seconds * 1000).toISOString().substring(14, 19)
}

function setSegments(segments) {
	if (segments == null) {
		setContent('Detecting sponsor segments...')
	} else if (segments.length == 0) {
		document.getElementById('content').innerHTML = '<div>No sponsored segments detected</div>'
	} else {
		document.getElementById('content').innerHTML = ''
		let list = '<ul>'
		segments.forEach(([start, end], id) => {
			list += '<li>' +
				'<a href="#" id="start-' + id + '">' + secondsToMMSS(start) + '</a>' +
				' - ' +
				'<a href="#" id="end-' + id + '">' + secondsToMMSS(end) + '</a>' +
				'</li>'
		});
		list += '</ul>'

		setContent('<h5>Sponsored segments:</h5>' + list)

		segments.forEach((segment, id) => {
			['start', 'end'].forEach((x, i) => {
				let element = document.getElementById(x + '-' + id)
				element.onclick = () => {
					chrome.tabs.query({
						active: true,
						currentWindow: true
					}, tabs => {
						chrome.tabs.sendMessage(
							tabs[0].id,
							{ from: 'popup', subject: 'seekTo', to: segment[i] }
						);
					});
				}
			})
		})
	}

}


function setTitle(title) {
	document.getElementById('video_title').innerHTML = title
}

function setDOMInfo(response) {
	if (response) {
		setTitle(response.title)
		setSegments(response.segments)
	} else {
		setContent('')
		setTitle('No YouTube video opened.')
	}
}

window.onload = function () {
	setContent('Loading...')
	setInterval(() => {
		chrome.tabs.query({
			active: true,
			currentWindow: true
		}, tabs => {
			// ...and send a request for the DOM info...

			chrome.tabs.sendMessage(
				tabs[0].id,
				{ from: 'popup', subject: 'getSegments' },
				// ...also specifying a callback to be called 
				//    from the receiving end (content script).
				setDOMInfo);
		});
	}, 1000)
};
