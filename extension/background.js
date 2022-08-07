const LAMBDA_URL = "https://gkbwcqkoc3f5wyv7ydb5si2koe0xogku.lambda-url.eu-west-2.on.aws"


chrome.tabs.onUpdated.addListener(
	async (tabId, changeInfo, tab) => {
		if (changeInfo.status == 'complete' && tab.active) {
			let matches = tab.url.match(/www\.youtube\.com\/watch\?.*v\=(.+)/)
			if (matches) {
				let video_id = matches[1]
				chrome.tabs.sendMessage(tabId, { segments: null, video_id });

				const controller = new AbortController();

				const timeoutId = setTimeout(() => controller.abort(), 60000);
				let response;
				response = await fetch(LAMBDA_URL + '/?video_id=' + video_id, { signal: controller.signal })

				clearTimeout(timeoutId)
				let body;

				body = await response.json()

				chrome.tabs.sendMessage(tabId, { segments: body, video_id });

				console.log({ segments: body, video_id })

			}

		}
	}
);