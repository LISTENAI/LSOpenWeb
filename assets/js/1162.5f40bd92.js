(self.webpackChunkls_docs_web=self.webpackChunkls_docs_web||[]).push([[1162],{895:function(e,t,n){"use strict";n.d(t,{dK:function(){return l},_k:function(){return o},vc:function(){return c},rx:function(){return i},qo:function(){return s},Hk:function(){return u},Iz:function(){return m}});var r=n(1336),a=n.n(r);n(892)(a()),n(4212).w(a()),n(4182)(a());var l=["en","zh"],o=!1,c=null,i=null,s=8,u=50,m={search_placeholder:"\u641c\u7d22",see_all_results:"\u67e5\u770b\u6240\u6709\u7ed3\u679c",no_results:"\u65e0\u67e5\u8be2\u7ed3\u679c",search_results_for:'\u641c\u7d22\u5173\u952e\u5b57 "{{ keyword }}" \u7684\u7ed3\u679c',count_documents_found_plural:"\u5171 {{ count }} \u4e2a\u67e5\u8be2\u7ed3\u679c",no_documents_were_found:"\u65e0\u67e5\u8be2\u7ed3\u679c",search_the_documentation:"Search the documentation",count_documents_found:"{{ count }} document found"}},4973:function(e,t,n){"use strict";n.d(t,{Z:function(){return m},I:function(){return u}});var r=n(7294),a=/{\w+}/g,l="{}";function o(e,t){var n=[],o=e.replace(a,(function(e){var a=e.substr(1,e.length-2),o=null==t?void 0:t[a];if(void 0!==o){var c=r.isValidElement(o)?o:String(o);return n.push(c),l}return e}));return 0===n.length?e:n.every((function(e){return"string"==typeof e}))?o.split(l).reduce((function(e,t,r){var a;return e.concat(t).concat(null!==(a=n[r])&&void 0!==a?a:"")}),""):o.split(l).reduce((function(e,t,a){return[].concat(e,[r.createElement(r.Fragment,{key:a},t,n[a])])}),[])}function c(e){return o(e.children,e.values)}var i=n(4644);function s(e){var t,n=e.id,r=e.message;return null!==(t=i[null!=n?n:r])&&void 0!==t?t:r}function u(e,t){var n,r=e.message;return o(null!==(n=s({message:r,id:e.id}))&&void 0!==n?n:r,t)}function m(e){var t,n=e.children,a=e.id,l=e.values,o=null!==(t=s({message:n,id:a}))&&void 0!==t?t:n;return r.createElement(c,{values:l},o)}},607:function(e,t,n){"use strict";var r=n(2122),a=n(9756),l=n(7294),o=["width","height","className"];t.Z=function(e){var t=e.width,n=void 0===t?30:t,c=e.height,i=void 0===c?30:c,s=e.className,u=(0,a.Z)(e,o);return l.createElement("svg",(0,r.Z)({className:s,width:n,height:i,viewBox:"0 0 30 30","aria-hidden":"true"},u),l.createElement("path",{stroke:"currentColor",strokeLinecap:"round",strokeMiterlimit:"10",strokeWidth:"2",d:"M4 7h22M4 15h22M4 23h22"}))}},9983:function(e,t,n){"use strict";n.d(t,{Z:function(){return Q}});var r=n(7294),a=n(6010),l=n(4973),o=n(9306),c="skipToContent_1knm";function i(e){e.setAttribute("tabindex","-1"),e.focus(),e.removeAttribute("tabindex")}var s=function(){var e=(0,r.useRef)(null);return(0,o.SL)((function(t){var n=t.location;e.current&&!n.hash&&i(e.current)})),r.createElement("div",{ref:e},r.createElement("a",{href:"#main",className:c,onClick:function(e){e.preventDefault();var t=document.querySelector("main:first-of-type")||document.querySelector(".main-wrapper");t&&i(t)}},r.createElement(l.Z,{id:"theme.common.skipToMainContent",description:"The skip to content label used for accessibility, allowing to rapidly navigate to main content with keyboard tab/enter navigation"},"Skip to main content")))},u="announcementBar_7rxc",m="announcementBarClose_2Qzf",d="announcementBarContent_zyhn",f="announcementBarCloseable_ykhy";var v=function(){var e,t=(0,o.nT)(),n=t.isClosed,c=t.close,i=(0,o.LU)().announcementBar;if(!i)return null;var s=i.content,v=i.backgroundColor,h=i.textColor,g=i.isCloseable;return!s||g&&n?null:r.createElement("div",{className:u,style:{backgroundColor:v,color:h},role:"banner"},r.createElement("div",{className:(0,a.Z)(d,(e={},e[f]=g,e)),dangerouslySetInnerHTML:{__html:s}}),g?r.createElement("button",{type:"button",className:(0,a.Z)(m,"clean-btn"),onClick:c,"aria-label":(0,l.I)({id:"theme.AnnouncementBar.closeButtonAriaLabel",message:"Close",description:"The ARIA label for close button of announcement bar"})},r.createElement("span",{"aria-hidden":"true"},"\xd7")):null)},h=n(7919),g=n(2122),p=n(4146),E=n(5202),b=n(9756),_=n(6742),Z=n(4996),k=n(5977),N=n(3919),w=["activeBasePath","activeBaseRegex","to","href","label","activeClassName","prependBaseUrlToHref"],y=["items","position","className"],C=["className"],L=["items","className","position"],I=["className"],D=["mobile"];function T(e){var t=e.activeBasePath,n=e.activeBaseRegex,a=e.to,l=e.href,o=e.label,c=e.activeClassName,i=void 0===c?"subnavbar__link--active":c,s=e.prependBaseUrlToHref,u=(0,b.Z)(e,w),m=(0,Z.Z)(a),d=(0,Z.Z)(t),f=(0,Z.Z)(l,{forcePrependBaseUrl:!0}),v=o&&l&&!(0,N.Z)(l);return r.createElement(_.Z,(0,g.Z)({},l?{href:s?f:l}:Object.assign({isNavLink:!0,activeClassName:i,to:m},t||n?{isActive:function(e,t){return n?new RegExp(n).test(t.pathname):t.pathname.startsWith(d)}}:null),u),v?r.createElement("span",{className:"navbar__exlink"},o):o)}function x(e){var t,n=e.items,l=e.position,o=e.className,c=(0,b.Z)(e,y),i=(0,r.useRef)(null),s=(0,r.useRef)(null),u=(0,r.useState)(!1),m=u[0],d=u[1];(0,r.useEffect)((function(){var e=function(e){i.current&&!i.current.contains(e.target)&&d(!1)};return document.addEventListener("mousedown",e),document.addEventListener("touchstart",e),function(){document.removeEventListener("mousedown",e),document.removeEventListener("touchstart",e)}}),[i]);var f=function(e,t){return void 0===t&&(t=!1),(0,a.Z)({"subnavbar__item subnavbar__link":!t,dropdown__link:t},e)};return n?r.createElement("div",{ref:i,className:(0,a.Z)("navbar__item","dropdown","dropdown--hoverable",{"dropdown--left":"left"===l,"dropdown--right":"right"===l,"dropdown--show":m})},r.createElement(T,(0,g.Z)({className:f(o)},c,{onClick:c.to?void 0:function(e){return e.preventDefault()},onKeyDown:function(e){"Enter"===e.key&&(e.preventDefault(),d(!m))}}),null!=(t=c.children)?t:c.label),r.createElement("ul",{ref:s,className:"dropdown__menu"},n.map((function(e,t){var a=e.className,l=(0,b.Z)(e,C);return r.createElement("li",{key:t},r.createElement(T,(0,g.Z)({onKeyDown:function(e){if(t===n.length-1&&"Tab"===e.key){e.preventDefault(),d(!1);var r=i.current.nextElementSibling;r&&r.focus()}},activeClassName:"dropdown__link--active",className:f(a,!0)},l)))})))):r.createElement(T,(0,g.Z)({className:f(o)},c))}function A(e){var t,n,l,c=e.items,i=e.className,s=(e.position,(0,b.Z)(e,L)),u=(0,r.useRef)(null),m=(0,k.TH)().pathname,d=(0,r.useState)((function(){var e;return null==(e=!(null!=c&&c.some((function(e){return(0,o.Mg)(e.to,m)}))))||e})),f=d[0],v=d[1],h=function(e,t){return void 0===t&&(t=!1),(0,a.Z)("menu__link",{"menu__link--sublist":t},e)};if(!c)return r.createElement("li",{className:"menu__list-item"},r.createElement(T,(0,g.Z)({className:h(i)},s)));var p=null!=(t=u.current)&&t.scrollHeight?(null==(n=u.current)?void 0:n.scrollHeight)+"px":void 0;return r.createElement("li",{className:(0,a.Z)("menu__list-item",{"menu__list-item--collapsed":f})},r.createElement(T,(0,g.Z)({role:"button",className:h(i,!0)},s,{onClick:function(e){e.preventDefault(),v((function(e){return!e}))}}),null!=(l=s.children)?l:s.label),r.createElement("ul",{className:"menu__list",ref:u,style:{height:f?void 0:p}},c.map((function(e,t){var n=e.className,a=(0,b.Z)(e,I);return r.createElement("li",{className:"menu__list-item",key:t},r.createElement(T,(0,g.Z)({activeClassName:"menu__link--active",className:h(n)},a,{onClick:s.onClick})))}))))}var B=function(e){var t=e.mobile,n=void 0!==t&&t,a=(0,b.Z)(e,D),l=n?A:x;return r.createElement(l,a)};var S=function(){var e=(0,o.LU)().subNavbar,t=(0,k.TH)(),n=(0,r.useState)(!1),l=n[0],c=n[1];(0,p.Z)(l);var i=(0,E.Z)();return(0,r.useEffect)((function(){i===E.D.desktop&&c(!1)}),[i]),r.createElement(r.Fragment,null,r.createElement("nav",{className:(0,a.Z)("navbar","subnavbar--fixed-top","sub-navbar")},r.createElement("div",{className:"navbar__inner subnavbar__inner"},r.createElement("div",{className:"navbar__items"},e.map((function(e){return t.pathname.indexOf(e.dirName||"")>=0?e.items.map((function(t,n){return r.createElement(B,(0,g.Z)({dirname:e.dirName},t,{key:n}))})):r.createElement(r.Fragment,null)}))))))},H="footerLogoLink_2hhA",U=n(2572),R=["to","href","label","prependBaseUrlToHref"];function M(e){var t=e.to,n=e.href,a=e.label,l=e.prependBaseUrlToHref,o=(0,b.Z)(e,R),c=(0,Z.Z)(t),i=(0,Z.Z)(n,{forcePrependBaseUrl:!0});return r.createElement(_.Z,(0,g.Z)({className:"footer__link-item"},n?{href:l?i:n}:{to:c},o),a)}var P=function(e){var t=e.sources,n=e.alt;return r.createElement(U.Z,{className:"footer__logo",alt:n,sources:t})};var V=function(){var e=(0,o.LU)().footer,t=e||{},n=t.copyright,l=t.links,c=void 0===l?[]:l,i=t.logo,s=void 0===i?{}:i,u={light:(0,Z.Z)(s.src),dark:(0,Z.Z)(s.srcDark||s.src)};return e?r.createElement("footer",{className:(0,a.Z)("footer",{"footer--dark":"dark"===e.style})},r.createElement("div",{className:"container"},c&&c.length>0&&r.createElement("div",{className:"row footer__links"},c.map((function(e,t){return r.createElement("div",{key:t,className:"col footer__col"},null!=e.title?r.createElement("div",{className:"footer__title"},e.title):null,null!=e.items&&Array.isArray(e.items)&&e.items.length>0?r.createElement("ul",{className:"footer__items"},e.items.map((function(e,t){return e.html?r.createElement("li",{key:t,className:"footer__item",dangerouslySetInnerHTML:{__html:e.html}}):r.createElement("li",{key:e.href||e.to,className:"footer__item"},r.createElement(M,e))}))):null)}))),(s||n)&&r.createElement("div",{className:"footer__bottom text--center"},s&&(s.src||s.srcDark)&&r.createElement("div",{className:"margin-bottom--sm"},s.href?r.createElement(_.Z,{href:s.href,className:H},r.createElement(P,{alt:s.alt,sources:u})):r.createElement(P,{alt:s.alt,sources:u})),n?r.createElement("div",{className:"footer__copyright",dangerouslySetInnerHTML:{__html:n}}):null))):null},W=n(5903),z=n(9105),F=n(2263);function O(e){var t=e.locale,n=e.version,a=e.tag;return r.createElement(z.Z,null,t&&r.createElement("meta",{name:"docusaurus_locale",content:t}),n&&r.createElement("meta",{name:"docusaurus_version",content:n}),a&&r.createElement("meta",{name:"docusaurus_tag",content:a}))}var j=n(3322);function G(){var e=(0,F.Z)().i18n,t=e.defaultLocale,n=e.locales,a=(0,o.l5)();return r.createElement(z.Z,null,n.map((function(e){return r.createElement("link",{key:e,rel:"alternate",href:a.createUrl({locale:e,fullyQualified:!0}),hrefLang:e})})),r.createElement("link",{rel:"alternate",href:a.createUrl({locale:t,fullyQualified:!0}),hrefLang:"x-default"}))}function K(e){var t=e.permalink,n=(0,F.Z)().siteConfig.url,a=function(){var e=(0,F.Z)().siteConfig.url,t=(0,k.TH)().pathname;return e+(0,Z.Z)(t)}(),l=t?""+n+t:a;return r.createElement(z.Z,null,r.createElement("meta",{property:"og:url",content:l}),r.createElement("link",{rel:"canonical",href:l}))}function q(e){var t=(0,F.Z)(),n=t.siteConfig,a=n.favicon,l=n.themeConfig,c=l.metadatas,i=l.image,s=t.i18n,u=s.currentLocale,m=s.localeConfigs,d=e.title,f=e.description,v=e.image,h=e.keywords,p=e.searchMetadatas,E=(0,Z.Z)(a),b=(0,o.pe)(d),_=u,k=m[u].direction;return r.createElement(r.Fragment,null,r.createElement(z.Z,null,r.createElement("html",{lang:_,dir:k}),a&&r.createElement("link",{rel:"shortcut icon",href:E}),r.createElement("title",null,b),r.createElement("meta",{property:"og:title",content:b}),v||i&&r.createElement("meta",{name:"twitter:card",content:"summary_large_image"})),r.createElement(j.Z,{description:f,keywords:h,image:v}),r.createElement(K,null),r.createElement(G,null),r.createElement(O,(0,g.Z)({tag:o.HX,locale:u},p)),r.createElement(z.Z,null,c.map((function(e,t){return r.createElement("meta",(0,g.Z)({key:"metadata_"+t},e))}))))}var J=function(){(0,r.useEffect)((function(){var e="navigation-with-keyboard";function t(t){"keydown"===t.type&&"Tab"===t.key&&document.body.classList.add(e),"mousedown"===t.type&&document.body.classList.remove(e)}return document.addEventListener("keydown",t),document.addEventListener("mousedown",t),function(){document.body.classList.remove(e),document.removeEventListener("keydown",t),document.removeEventListener("mousedown",t)}}),[])};var Q=function(e){var t=e.children,n=e.noFooter,l=e.wrapperClassName,c=e.pageClassName;return J(),r.createElement(W.Z,null,r.createElement(q,e),r.createElement(s,null),r.createElement(v,null),r.createElement(h.Z,null),r.createElement(S,null),r.createElement("div",{className:(0,a.Z)(o.kM.wrapper.main,l,c)},t),!n&&r.createElement(V,null))}},5903:function(e,t,n){"use strict";n.d(t,{Z:function(){return E}});var r=n(7294),a=n(412),l=n(9306),o=(0,l.WA)("theme"),c="light",i="dark",s=function(e){return e===i?i:c},u=function(e){(0,l.WA)("theme").set(s(e))},m=function(){var e=(0,l.LU)().colorMode,t=e.defaultMode,n=e.disableSwitch,m=e.respectPrefersColorScheme,d=(0,r.useState)(function(e){return a.Z.canUseDOM?s(document.documentElement.getAttribute("data-theme")):s(e)}(t)),f=d[0],v=d[1],h=(0,r.useCallback)((function(){v(c),u(c)}),[]),g=(0,r.useCallback)((function(){v(i),u(i)}),[]);return(0,r.useEffect)((function(){document.documentElement.setAttribute("data-theme",s(f))}),[f]),(0,r.useEffect)((function(){if(!n)try{var e=o.get();null!==e&&v(s(e))}catch(t){console.error(t)}}),[v]),(0,r.useEffect)((function(){n&&!m||window.matchMedia("(prefers-color-scheme: dark)").addListener((function(e){var t=e.matches;v(t?i:c)}))}),[]),{isDarkTheme:f===i,setLightTheme:h,setDarkTheme:g}},d=n(7526);var f=function(e){var t=m(),n=t.isDarkTheme,a=t.setLightTheme,l=t.setDarkTheme;return r.createElement(d.Z.Provider,{value:{isDarkTheme:n,setLightTheme:a,setDarkTheme:l}},e.children)},v="docusaurus.tab.",h=function(){var e=(0,r.useState)({}),t=e[0],n=e[1],a=(0,r.useCallback)((function(e,t){(0,l.WA)("docusaurus.tab."+e).set(t)}),[]);return(0,r.useEffect)((function(){try{var e={};(0,l._f)().forEach((function(t){if(t.startsWith(v)){var n=t.substring(v.length);e[n]=(0,l.WA)(t).get()}})),n(e)}catch(t){console.error(t)}}),[]),{tabGroupChoices:t,setTabGroupChoices:function(e,t){n((function(n){var r;return Object.assign({},n,((r={})[e]=t,r))})),a(e,t)}}},g=(0,r.createContext)(void 0);var p=function(e){var t=h(),n=t.tabGroupChoices,a=t.setTabGroupChoices;return r.createElement(g.Provider,{value:{tabGroupChoices:n,setTabGroupChoices:a}},e.children)};function E(e){var t=e.children;return r.createElement(f,null,r.createElement(l.pl,null,r.createElement(p,null,r.createElement(l.L5,null,t))))}},5475:function(e,t,n){"use strict";var r=n(2122),a=n(9756),l=n(7294),o=n(6742),c=n(2572),i=n(4996),s=n(2263),u=["imageClassName","titleClassName"];t.Z=function(e){var t=(0,s.Z)(),n=t.siteConfig,m=n.title,d=n.themeConfig.navbar,f=d.title,v=d.logo,h=void 0===v?{src:""}:v,g=t.isClient,p=e.imageClassName,E=e.titleClassName,b=(0,a.Z)(e,u),_=(0,i.Z)(h.href||"/"),Z={light:(0,i.Z)(h.src),dark:(0,i.Z)(h.srcDark||h.src)};return l.createElement(o.Z,(0,r.Z)({to:_},b,h.target&&{target:h.target}),h.src&&l.createElement(c.Z,{key:g,className:p,sources:Z,alt:h.alt||f||m}),null!=f&&l.createElement("b",{className:E},f))}},7919:function(e,t,n){"use strict";n.d(t,{Z:function(){return M}});var r=n(2122),a=n(7294),l=n(6010),o=n(5865),c=n(2572),i=n(9306),s=n(4996),u=n(2263),m={toggle:"toggle_ZCuN"},d=function(e){var t=e.icon,n=e.style;return a.createElement("span",{className:(0,l.Z)(m.toggle,m.dark),style:n},a.createElement(c.Z,{sources:t}))},f=function(e){var t=e.icon,n=e.style;return a.createElement("span",{className:(0,l.Z)(m.toggle,m.light),style:n},a.createElement(c.Z,{sources:t}))},v=(0,a.memo)((function(e){var t=e.className,n=e.icons,r=e.checked,o=e.disabled,c=e.onChange,i=(0,a.useState)(r),s=i[0],u=i[1],m=(0,a.useRef)(null);return a.createElement("div",{className:(0,l.Z)("react-toggle",t,{"react-toggle--checked":s,"react-toggle--disabled":o})},a.createElement("div",{className:"react-toggle-track",role:"button",tabIndex:-1,onClick:function(){var e;return null==(e=m.current)?void 0:e.click()}},a.createElement("div",{className:"react-toggle-track-check"},n.checked),a.createElement("div",{className:"react-toggle-track-x"},n.unchecked)),a.createElement("input",{ref:m,checked:s,type:"checkbox",className:"react-toggle-screenreader-only","aria-label":"Switch between dark and light mode",onChange:c,onClick:function(){return u(!s)}}))}));function h(e){var t=(0,i.LU)().colorMode.switchConfig,n=t.darkIcon,l=t.darkIconStyle,o=t.lightIcon,c=t.lightIconStyle,m=(0,u.Z)().isClient;return a.createElement(v,(0,r.Z)({disabled:!m,icons:{checked:a.createElement(f,{icon:{light:(0,s.Z)(o.light),dark:(0,s.Z)(o.dark)},style:c}),unchecked:a.createElement(d,{icon:{light:(0,s.Z)(n.light),dark:(0,s.Z)(n.dark)},style:l})}},e))}var g=n(7181),p=n(5977),E=n(9936),b=function(e){var t=(0,p.TH)(),n=(0,a.useState)(e),r=n[0],l=n[1],o=(0,a.useRef)(!1),c=(0,a.useState)(0),s=c[0],u=c[1],m=(0,a.useCallback)((function(e){null!==e&&u(e.getBoundingClientRect().height)}),[]);return(0,E.Z)((function(t,n){var r=t.scrollY,a=n.scrollY;if(e)if(r<s)l(!0);else{if(o.current)return o.current=!1,void l(!1);a&&0===r&&l(!0);var c=document.documentElement.scrollHeight-s,i=window.innerHeight;a&&r>=a?l(!1):r+i<c&&l(!0)}}),[s,o]),(0,i.SL)((function(t){e&&!t.location.hash&&l(!0)})),(0,a.useEffect)((function(){e&&t.hash&&(o.current=!0)}),[t.hash]),{navbarRef:m,isNavbarVisible:r}},_=n(4146),Z=n(5202),k=n(9756),N=n(4938),w=["width","height"],y=function(e){var t=e.width,n=void 0===t?20:t,l=e.height,o=void 0===l?20:l,c=(0,k.Z)(e,w);return a.createElement("svg",(0,r.Z)({viewBox:"0 0 20 20",width:n,height:o,"aria-hidden":"true"},c),a.createElement("path",{fill:"currentColor",d:"M19.753 10.909c-.624-1.707-2.366-2.726-4.661-2.726-.09 0-.176.002-.262.006l-.016-2.063 3.525-.607c.115-.019.133-.119.109-.231-.023-.111-.167-.883-.188-.976-.027-.131-.102-.127-.207-.109-.104.018-3.25.461-3.25.461l-.013-2.078c-.001-.125-.069-.158-.194-.156l-1.025.016c-.105.002-.164.049-.162.148l.033 2.307s-3.061.527-3.144.543c-.084.014-.17.053-.151.143.019.09.19 1.094.208 1.172.018.08.072.129.188.107l2.924-.504.035 2.018c-1.077.281-1.801.824-2.256 1.303-.768.807-1.207 1.887-1.207 2.963 0 1.586.971 2.529 2.328 2.695 3.162.387 5.119-3.06 5.769-4.715 1.097 1.506.256 4.354-2.094 5.98-.043.029-.098.129-.033.207l.619.756c.08.096.206.059.256.023 2.51-1.73 3.661-4.515 2.869-6.683zm-7.386 3.188c-.966-.121-.944-.914-.944-1.453 0-.773.327-1.58.876-2.156a3.21 3.21 0 011.229-.799l.082 4.277a2.773 2.773 0 01-1.243.131zm2.427-.553l.046-4.109c.084-.004.166-.01.252-.01.773 0 1.494.145 1.885.361.391.217-1.023 2.713-2.183 3.758zm-8.95-7.668a.196.196 0 00-.196-.145h-1.95a.194.194 0 00-.194.144L.008 16.916c-.017.051-.011.076.062.076h1.733c.075 0 .099-.023.114-.072l1.008-3.318h3.496l1.008 3.318c.016.049.039.072.113.072h1.734c.072 0 .078-.025.062-.076-.014-.05-3.083-9.741-3.494-11.04zm-2.618 6.318l1.447-5.25 1.447 5.25H3.226z"}))},C=["mobile","dropdownItemsBefore","dropdownItemsAfter"];function L(e){var t=e.mobile,n=e.dropdownItemsBefore,l=e.dropdownItemsAfter,o=(0,k.Z)(e,C),c=(0,u.Z)().i18n,s=c.currentLocale,m=c.locales,d=c.localeConfigs,f=(0,i.l5)();function v(e){return d[e].label}var h=m.map((function(e){var t="pathname://"+f.createUrl({locale:e,fullyQualified:!1});return{isNavLink:!0,label:v(e),to:t,target:"_self",autoAddBaseUrl:!1,className:e===s?"dropdown__link--active":"",style:{textTransform:"capitalize"}}})),g=[].concat(n,h,l),p=t?"Languages":v(s);return a.createElement(N.Z,(0,r.Z)({},o,{href:"#",mobile:t,label:a.createElement("span",null,a.createElement(y,{style:{verticalAlign:"text-bottom",marginRight:5}}),a.createElement("span",null,p)),items:g}))}function I(e){return e.mobile?null:a.createElement(o.Z,null)}var D=["type"],T={default:function(){return N.Z},localeDropdown:function(){return L},search:function(){return I},docsVersion:function(){return n(3747).Z},docsVersionDropdown:function(){return n(6701).Z},doc:function(){return n(6431).Z}};function x(e){var t=e.type,n=(0,k.Z)(e,D),r=function(e){void 0===e&&(e="default");var t=T[e];if(!t)throw new Error('No NavbarItem component found for type "'+e+'".');return t()}(t);return a.createElement(r,n)}var A=n(5475),B=n(607),S="displayOnlyInLargeViewport_3VZa",H="navbarHideable_T8CR",U="navbarHidden_1tRw",R="right";var M=function(){var e,t=(0,i.LU)(),n=t.navbar,c=n.items,s=n.hideOnScroll,u=n.style,m=t.colorMode.disableSwitch,d=(0,a.useState)(!1),f=d[0],v=d[1],p=(0,g.Z)(),E=p.isDarkTheme,k=p.setLightTheme,N=p.setDarkTheme,w=b(s),y=w.navbarRef,C=w.isNavbarVisible;(0,_.Z)(f);var L=(0,a.useCallback)((function(){v(!0)}),[v]),I=(0,a.useCallback)((function(){v(!1)}),[v]),D=(0,a.useCallback)((function(e){return e.target.checked?N():k()}),[k,N]),T=(0,Z.Z)();(0,a.useEffect)((function(){T===Z.D.desktop&&v(!1)}),[T]);var M=c.some((function(e){return"search"===e.type})),P=function(e){return{leftItems:e.filter((function(e){var t;return"left"===(null!=(t=e.position)?t:R)})),rightItems:e.filter((function(e){var t;return"right"===(null!=(t=e.position)?t:R)}))}}(c),V=P.leftItems,W=P.rightItems;return a.createElement("nav",{ref:y,className:(0,l.Z)("navbar","navbar--fixed-top",(e={"navbar--dark":"dark"===u,"navbar--primary":"primary"===u,"navbar-sidebar--show":f},e[H]=s,e[U]=s&&!C,e))},a.createElement("div",{className:"navbar__inner"},a.createElement("div",{className:"navbar__items"},null!=c&&0!==c.length&&a.createElement("button",{"aria-label":"Navigation bar toggle",className:"navbar__toggle clean-btn",type:"button",tabIndex:0,onClick:L,onKeyDown:L},a.createElement(B.Z,null)),a.createElement(A.Z,{className:"navbar__brand",imageClassName:"navbar__logo",titleClassName:"navbar__title"}),V.map((function(e,t){return a.createElement(x,(0,r.Z)({},e,{key:t}))}))),a.createElement("div",{className:"navbar__items navbar__items--right"},W.map((function(e,t){return a.createElement(x,(0,r.Z)({},e,{key:t}))})),!m&&a.createElement(h,{className:S,checked:E,onChange:D}),!M&&a.createElement(o.Z,null))),a.createElement("div",{role:"presentation",className:"navbar-sidebar__backdrop",onClick:I}),a.createElement("div",{className:"navbar-sidebar"},a.createElement("div",{className:"navbar-sidebar__brand"},a.createElement(A.Z,{className:"navbar__brand",imageClassName:"navbar__logo",titleClassName:"navbar__title",onClick:I}),!m&&f&&a.createElement(h,{checked:E,onChange:D})),a.createElement("div",{className:"navbar-sidebar__items"},a.createElement("div",{className:"menu"},a.createElement("ul",{className:"menu__list"},c.map((function(e,t){return a.createElement(x,(0,r.Z)({mobile:!0},e,{onClick:I,key:t}))})))))))}},4938:function(e,t,n){"use strict";var r=n(2122),a=n(9756),l=n(7294),o=n(6010),c=n(6742),i=n(4996),s=n(5977),u=n(9306),m=n(3919),d=["activeBasePath","activeBaseRegex","to","href","label","activeClassName","prependBaseUrlToHref"],f=["items","position","className"],v=["className"],h=["items","className","position"],g=["className"],p=["mobile"];function E(e){var t=e.activeBasePath,n=e.activeBaseRegex,o=e.to,s=e.href,u=e.label,f=e.activeClassName,v=void 0===f?"navbar__link--active":f,h=e.prependBaseUrlToHref,g=(0,a.Z)(e,d),p=(0,i.Z)(o),E=(0,i.Z)(s,{forcePrependBaseUrl:!0}),b=u&&s&&!(0,m.Z)(s);return l.createElement(c.Z,(0,r.Z)({},s?{href:h?E:s}:Object.assign({isNavLink:!0,activeClassName:v,to:p},t||n?{isActive:function(e,r){return n?new RegExp(n).test(r.pathname):r.pathname.startsWith(t)}}:null),g),b?l.createElement("span",{className:"navbar__exlink"},u):u)}function b(e){var t,n=e.items,c=e.position,i=e.className,s=(0,a.Z)(e,f),u=(0,l.useRef)(null),m=(0,l.useRef)(null),d=(0,l.useState)(!1),h=d[0],g=d[1];(0,l.useEffect)((function(){var e=function(e){u.current&&!u.current.contains(e.target)&&g(!1)};return document.addEventListener("mousedown",e),document.addEventListener("touchstart",e),function(){document.removeEventListener("mousedown",e),document.removeEventListener("touchstart",e)}}),[u]);var p=function(e,t){return void 0===t&&(t=!1),(0,o.Z)({"navbar__item navbar__link":!t,dropdown__link:t},e)};return n?l.createElement("div",{ref:u,className:(0,o.Z)("navbar__item","dropdown","dropdown--hoverable",{"dropdown--left":"left"===c,"dropdown--right":"right"===c,"dropdown--show":h})},l.createElement(E,(0,r.Z)({className:p(i)},s,{onClick:s.to?void 0:function(e){return e.preventDefault()},onKeyDown:function(e){"Enter"===e.key&&(e.preventDefault(),g(!h))}}),null!=(t=s.children)?t:s.label),l.createElement("ul",{ref:m,className:"dropdown__menu"},n.map((function(e,t){var o=e.className,c=(0,a.Z)(e,v);return l.createElement("li",{key:t},l.createElement(E,(0,r.Z)({onKeyDown:function(e){if(t===n.length-1&&"Tab"===e.key){e.preventDefault(),g(!1);var r=u.current.nextElementSibling;r&&r.focus()}},activeClassName:"dropdown__link--active",className:p(o,!0)},c)))})))):l.createElement(E,(0,r.Z)({className:p(i)},s))}function _(e){var t,n,c,i=e.items,m=e.className,d=(e.position,(0,a.Z)(e,h)),f=(0,l.useRef)(null),v=(0,s.TH)().pathname,p=(0,l.useState)((function(){var e;return null==(e=!(null!=i&&i.some((function(e){return(0,u.Mg)(e.to,v)}))))||e})),b=p[0],_=p[1],Z=function(e,t){return void 0===t&&(t=!1),(0,o.Z)("menu__link",{"menu__link--sublist":t},e)};if(!i)return l.createElement("li",{className:"menu__list-item"},l.createElement(E,(0,r.Z)({activeClassName:"mobile__navbar__link--active",className:Z(m)},d)));var k=null!=(t=f.current)&&t.scrollHeight?(null==(n=f.current)?void 0:n.scrollHeight)+"px":void 0;return l.createElement("li",{className:(0,o.Z)("menu__list-item",{"menu__list-item--collapsed":b})},l.createElement(E,(0,r.Z)({role:"button",className:Z(m,!0)},d,{onClick:function(e){e.preventDefault(),_((function(e){return!e}))}}),null!=(c=d.children)?c:d.label),l.createElement("ul",{className:"menu__list",ref:f,style:{height:b?void 0:k}},i.map((function(e,t){var n=e.className,o=(0,a.Z)(e,g);return l.createElement("li",{className:"menu__list-item",key:t},l.createElement(E,(0,r.Z)({activeClassName:"menu__link--active",className:Z(n)},o,{onClick:d.onClick})))}))))}t.Z=function(e){var t=e.mobile,n=void 0!==t&&t,r=(0,a.Z)(e,p),o=n?_:b;return l.createElement(o,r)}},6431:function(e,t,n){"use strict";n.d(t,{Z:function(){return d}});var r=n(2122),a=n(9756),l=n(7294),o=n(4938),c=n(7227),i=n(6010),s=n(9306),u=n(8780),m=["docId","activeSidebarClassName","label","docsPluginId"];function d(e){var t,n=e.docId,d=e.activeSidebarClassName,f=e.label,v=e.docsPluginId,h=(0,a.Z)(e,m),g=(0,c.Iw)(v),p=g.activeVersion,E=g.activeDoc,b=(0,s.J)(v).preferredVersion,_=(0,c.yW)(v),Z=function(e,t){var n,r=(n=[]).concat.apply(n,e.map((function(e){return e.docs}))),a=r.find((function(e){return e.id===t}));if(!a){var l=r.map((function(e){return e.id})).join("\n- ");throw new Error("DocNavbarItem: couldn't find any doc with id \""+t+'" in version'+(e.length?"s":"")+" "+e.map((function(e){return e.name})).join(", ")+'".\nAvailable doc ids are:\n- '+l)}return a}((0,u.uniq)([p,b,_].filter(Boolean)),n);return l.createElement(o.Z,(0,r.Z)({exact:!0},h,{className:(0,i.Z)(h.className,(t={},t[d]=E&&E.sidebar===Z.sidebar,t)),label:null!=f?f:Z.id,to:Z.path}))}},6701:function(e,t,n){"use strict";n.d(t,{Z:function(){return m}});var r=n(2122),a=n(9756),l=n(7294),o=n(4938),c=n(7227),i=n(9306),s=["mobile","docsPluginId","dropdownActiveClassDisabled","dropdownItemsBefore","dropdownItemsAfter"],u=function(e){return e.docs.find((function(t){return t.id===e.mainDocId}))};function m(e){var t,n,m=e.mobile,d=e.docsPluginId,f=e.dropdownActiveClassDisabled,v=e.dropdownItemsBefore,h=e.dropdownItemsAfter,g=(0,a.Z)(e,s),p=(0,c.Iw)(d),E=(0,c.gB)(d),b=(0,c.yW)(d),_=(0,i.J)(d),Z=_.preferredVersion,k=_.savePreferredVersionName;var N=function(){var e=E.map((function(e){var t=(null==p?void 0:p.alternateDocVersions[e.name])||u(e);return{isNavLink:!0,label:e.label,to:t.path,isActive:function(){return e===(null==p?void 0:p.activeVersion)},onClick:function(){k(e.name)}}})),t=[].concat(v,e,h);if(!(t.length<=1))return t}(),w=null!=(t=null!=(n=p.activeVersion)?n:Z)?t:b,y=m&&N?"Versions":w.label,C=m&&N?void 0:u(w).path;return l.createElement(o.Z,(0,r.Z)({},g,{mobile:m,label:y,to:C,items:N,isActive:f?function(){return!1}:void 0}))}},3747:function(e,t,n){"use strict";n.d(t,{Z:function(){return u}});var r=n(2122),a=n(9756),l=n(7294),o=n(4938),c=n(7227),i=n(9306),s=["label","to","docsPluginId"];function u(e){var t,n=e.label,u=e.to,m=e.docsPluginId,d=(0,a.Z)(e,s),f=(0,c.zu)(m),v=(0,i.J)(m).preferredVersion,h=(0,c.yW)(m),g=null!=(t=null!=f?f:v)?t:h,p=null!=n?n:g.label,E=null!=u?u:function(e){return e.docs.find((function(t){return t.id===e.mainDocId}))}(g).path;return l.createElement(o.Z,(0,r.Z)({},d,{label:p,to:E}))}},7818:function(e,t,n){"use strict";n.r(t);var r=n(7294),a=n(9983),l=n(4973);t.default=function(){return r.createElement(a.Z,{title:(0,l.I)({id:"theme.NotFound.title",message:"Page Not Found"})},r.createElement("main",{className:"container margin-vert--xl"},r.createElement("div",{className:"row"},r.createElement("div",{className:"col col--6 col--offset-3"},r.createElement("h1",{className:"hero__title"},r.createElement(l.Z,{id:"theme.NotFound.title",description:"The title of the 404 page"},"Page Not Found")),r.createElement("p",null,r.createElement(l.Z,{id:"theme.NotFound.p1",description:"The first paragraph of the 404 page"},"We could not find what you were looking for.")),r.createElement("p",null,r.createElement(l.Z,{id:"theme.NotFound.p2",description:"The 2nd paragraph of the 404 page"},"Please contact the owner of the site that linked you to the original URL and let them know their link is broken."))))))}},3322:function(e,t,n){"use strict";n.d(t,{Z:function(){return c}});var r=n(7294),a=n(9105),l=n(9306),o=n(4996);function c(e){var t=e.title,n=e.description,c=e.keywords,i=e.image,s=(0,l.LU)().image,u=(0,l.pe)(t),m=(0,o.Z)(i||s,{absolute:!0});return r.createElement(a.Z,null,t&&r.createElement("title",null,u),t&&r.createElement("meta",{property:"og:title",content:u}),n&&r.createElement("meta",{name:"description",content:n}),n&&r.createElement("meta",{property:"og:description",content:n}),c&&r.createElement("meta",{name:"keywords",content:Array.isArray(c)?c.join(","):c}),m&&r.createElement("meta",{property:"og:image",content:m}),m&&r.createElement("meta",{name:"twitter:image",content:m}))}},7526:function(e,t,n){"use strict";var r=n(7294).createContext(void 0);t.Z=r},2572:function(e,t,n){"use strict";n.d(t,{Z:function(){return m}});var r=n(2122),a=n(9756),l=n(7294),o=n(6010),c=n(2263),i=n(7181),s={themedImage:"themedImage_1WWj","themedImage--light":"themedImage--light_1H0o","themedImage--dark":"themedImage--dark_2hTa"},u=["sources","className","alt"],m=function(e){var t=(0,c.Z)().isClient,n=(0,i.Z)().isDarkTheme,m=e.sources,d=e.className,f=e.alt,v=void 0===f?"":f,h=(0,a.Z)(e,u),g=t?n?["dark"]:["light"]:["light","dark"];return l.createElement(l.Fragment,null,g.map((function(e){return l.createElement("img",(0,r.Z)({key:e,src:m[e],alt:v,className:(0,o.Z)(s.themedImage,s["themedImage--"+e],d)},h))})))}},7227:function(e,t,n){"use strict";n.d(t,{Iw:function(){return r.useActiveDocContext},gA:function(){return r.useActivePlugin},zu:function(){return r.useActiveVersion},_r:function(){return r.useAllDocsData},Jo:function(){return r.useDocVersionSuggestions},zh:function(){return r.useDocsData},yW:function(){return r.useLatestVersion},gB:function(){return r.useVersions}});var r=n(6730)},4146:function(e,t,n){"use strict";var r=n(7294);t.Z=function(e){void 0===e&&(e=!0),(0,r.useEffect)((function(){return document.body.style.overflow=e?"hidden":"visible",function(){document.body.style.overflow="visible"}}),[e])}},9936:function(e,t,n){"use strict";var r=n(7294),a=n(412),l=function(){return{scrollX:a.Z.canUseDOM?window.pageXOffset:0,scrollY:a.Z.canUseDOM?window.pageYOffset:0}};t.Z=function(e,t){void 0===t&&(t=[]);var n=(0,r.useRef)(l()),a=function(){var t=l();e&&e(t,n.current),n.current=t};(0,r.useEffect)((function(){var e={passive:!0};return a(),window.addEventListener("scroll",a,e),function(){return window.removeEventListener("scroll",a,e)}}),t)}},7181:function(e,t,n){"use strict";var r=n(7294),a=n(7526);t.Z=function(){var e=(0,r.useContext)(a.Z);if(null==e)throw new Error('"useThemeContext" is used outside of "Layout" component. Please see https://docusaurus.io/docs/api/themes/configuration#usethemecontext.');return e}},5202:function(e,t,n){"use strict";n.d(t,{D:function(){return l}});var r=n(7294),a=n(412),l={desktop:"desktop",mobile:"mobile"};t.Z=function(){var e=a.Z.canUseDOM;function t(){if(e)return window.innerWidth>1255?l.desktop:l.mobile}var n=(0,r.useState)(t),o=n[0],c=n[1];return(0,r.useEffect)((function(){if(e)return window.addEventListener("resize",n),function(){return window.removeEventListener("resize",n)};function n(){c(t())}}),[]),o}}}]);