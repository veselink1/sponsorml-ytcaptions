const player = document.getElementById('movie_player');

setInterval(() => {
	if (player.getPlayerState() == 1) {
		const event = new CustomEvent('currentVideoTime', { detail: player.getCurrentTime() })
		document.documentElement.dispatchEvent(event)
	}
}, 1000)

document.documentElement.addEventListener('playVideo', () => player.playVideo())
document.documentElement.addEventListener('pauseVideo', () => player.pauseVideo())
document.documentElement.addEventListener('seekTo', (e) => player.seekTo(e.detail))

