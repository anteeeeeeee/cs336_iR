//------------------------ Update result ------------------------//

// Cache for storing search results
const searchCache = new Map();

// AbortController for cancelling ongoing requests
let currentAbortController = null;

// Create a single IntersectionObserver instance
const imageObserver = new IntersectionObserver((entries, observer) => {
  entries.forEach(entry => {
      if (entry.isIntersecting) {
          const img = entry.target;
          img.src = img.dataset.src;
          observer.unobserve(img);
      }
  });
}, {
  rootMargin: '100px'
});


function createImageDiv(result, index) {
  frameInfo = `${result.entity.video}-${parseFloat(result.entity.time.toFixed(2))}`;
  // console.log(result)

  const div = document.createElement('div');
  div.className = 'img-dis';
    div.innerHTML = `
    <img alt="" class="result" loading="lazy" height="100px" id="${index}" src="/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIC/get_keyframes/data-batch-2/${result.entity.video}/keyframes/keyframe_${result.entity.frame_id}.webp">
    <div class="infor">${frameInfo}</div>
    <div name="similarity_search" class="similarity_search"></div>
    <div class="export_icon"></div>
  `;

  const img = div.querySelector('img');
  img.setAttribute('draggable', 'true');
  img.addEventListener('dragstart', drag);
  
  const exportIcon = div.querySelector('.export_icon');
  exportIcon.addEventListener('click', () => {
      img.dataset.src = "/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIC/get_keyframes/data-batch-2/"  + result.entity.video + "/keyframes/keyframe_" + result.entity.frame_id + ".webp";
    frameId = result.entity.time;
    addImageToExportArea(frameId, imagePath, frameInfo);
  });

  // // Using unified event listeners
  // div.addEventListener('mousedown', handleMiddleClick);
  // const exportIcon = div.querySelector('.export_icon');
  // exportIcon.addEventListener('click', handleExportIconClick);

  return div;
}


// Update UI with search results using IntersectionObserver for lazy loading
// Update the new right panel
function updateRightPanel_list(results) {
  const listPhoto = document.getElementById("list-photo");
  const fragment = document.createDocumentFragment();
  const existingDivs = Array.from(listPhoto.children);

  const updatedDivs = results.map((result, index) => {
      let div;
      if (index < existingDivs.length) {
          div = existingDivs[index];
          div.style.display = 'block';
      } else {
          div = createImageDiv(result, index + 1);
          fragment.appendChild(div);
      }

      const img = div.querySelector('img');
      const infor = div.querySelector('.infor');
      img.id = index + 1;

      img.dataset.src = "/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIC/get_keyframes/data-batch-2/"  + result.entity.video + "/keyframes/keyframe_" + result.entity.frame_id + ".webp";

      img.src = '';
      infor.textContent = `${result.entity.video}-${parseFloat(result.entity.time.toFixed(2))}`;

      imageObserver.observe(img);
      return div;
  });

  // Remove excess divs
  existingDivs.slice(results.length).forEach(div => div.remove());

  // Append new divs if any
  if (fragment.children.length > 0) {
      listPhoto.appendChild(fragment);
  }

  listPhoto.scrollTop = 0;

  return updatedDivs;
}



//-----------------------------------------------------------------------------------------------------//

// Get current results from the right panel
function getCurrentResults() {
  return Array.from(document.querySelectorAll('.img-dis')).map(div => ({
    entity: {
      path: div.querySelector('img').dataset.src,
      video: div.querySelector('.infor').textContent.split('-')[0],
      frame_id: div.querySelector('.infor').textContent.split('-')[1]
    },
    id: div.querySelector('img').id
  }));
}


// Group the results by video
function groupResultsByVideo(results) {
  return results.reduce((groups, result) => {
      const videoName = result.entity.video;
      (groups[videoName] = groups[videoName] || []).push(result);
      return groups;
  }, {});
}

// Update the new right panel
function updateRightPanel_rows(results) {
  const imagesRows = document.getElementById('images-rows');
  imagesRows.innerHTML = ''; // Clear existing content

  const videoGroups = groupResultsByVideo(results);
  
  const fragment = document.createDocumentFragment();

  Object.entries(videoGroups).forEach(([videoName, videoResults], index) => {
      const videoSection = document.createElement('div');
      videoSection.className = 'group-frame';

      if (index > 0) {
          videoSection.appendChild(document.createElement('hr'));
      }

      const videoTitle = document.createElement('h3');
      videoTitle.textContent = videoName;
      videoSection.appendChild(videoTitle);

      const resultFragment = document.createDocumentFragment();
      videoResults.forEach((result) => {
          const originalIndex = results.findIndex(r => r.id === result.id);
          const imgDiv = createImageDiv(result, originalIndex + 1);
          resultFragment.appendChild(imgDiv);
      });

      videoSection.appendChild(resultFragment);
      fragment.appendChild(videoSection);
  });

  imagesRows.appendChild(fragment);

  imagesRows.scrollTop = 0;
}

// Update both panels
function updateUIWithSearchResults(results) {
  updateRightPanel_list(results);
  updateRightPanel_rows(results);
  updateObjectList(results);
  
  // If the object list is visible, update its content
  if (isObjectListVisible) {
    renderObjectList();
    // Ensure the object list content remains visible
    const objectListContent = document.querySelector('.object-list-content');
    if (objectListContent) {
      objectListContent.classList.add('show');
    }
  }


  document.querySelector('.show-image-1').scrollTop = 0;
  document.querySelector('.show-image-2').scrollTop = 0;
  
  addImageEventListeners();
}


function addImageEventListeners() {
  document.querySelectorAll('.img-dis img').forEach(img => {
    img.addEventListener('dragstart', drag);
  });

  document.querySelectorAll('.export_icon').forEach(icon => {
    icon.addEventListener('click', (event) => {
      const container = event.target.closest('.img-dis');
      const img = container.querySelector('img');
      const infor = container.querySelector('.infor');
      const frameId = infor.textContent.split('-')[1];
      const imageSrc = img.src;
      addImageToExportArea(frameId, imageSrc, infor.textContent);
    });
  });

  // Add middle-click event listener to all relevant containers
  document.querySelectorAll('.img-dis, .frame-container, .preview-image-wrapper, .current-preview-wrapper').forEach(container => {
    container.addEventListener('mousedown', handleMiddleClick);
  });
}





  
function cleanupSearchResults() {
  // Clear main content areas
  ['list-photo', 'images-rows'].forEach(id => {
    const element = document.getElementById(id);
    if (element) {
      // Remove all observers and event listeners
      element.querySelectorAll('img').forEach(img => imageObserver.unobserve(img));
      element.innerHTML = '';
    }
  });

  // Clear object list if exists
  const objectList = document.querySelector('.object-list-content');
  if (objectList) {
    objectList.innerHTML = '';
    objectList.classList.remove('show');
  }

  // Reset scroll positions
  ['show-image-1', 'show-image-2'].forEach(className => {
    const element = document.querySelector(`.${className}`);
    if (element) element.scrollTop = 0;
  });
}