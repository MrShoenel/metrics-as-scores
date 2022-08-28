(async() => {
	const id = document.getElementById.bind(document)
	, sleep = ms => new Promise((res, rej) => {
		setTimeout(res, ms)
	})
	, asyncElem = async elemId => {
		while (id(elemId) === null) {
			await sleep(10);
		}
		return id(elemId);
	}
	, authors = await asyncElem('authors');

	authors.addEventListener('click', (elem, ev) => {
		window.alert('This application belongs to a paper currently under double-blind review. The authors cannot be revealed until after this process.')
	});
})()