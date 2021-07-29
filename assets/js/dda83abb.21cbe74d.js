(self.webpackChunkls_docs_web=self.webpackChunkls_docs_web||[]).push([[8466],{3905:function(e,t,n){"use strict";n.d(t,{Zo:function(){return p},kt:function(){return c}});var i=n(7294);function a(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function s(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);t&&(i=i.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,i)}return n}function l(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?s(Object(n),!0).forEach((function(t){a(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):s(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function r(e,t){if(null==e)return{};var n,i,a=function(e,t){if(null==e)return{};var n,i,a={},s=Object.keys(e);for(i=0;i<s.length;i++)n=s[i],t.indexOf(n)>=0||(a[n]=e[n]);return a}(e,t);if(Object.getOwnPropertySymbols){var s=Object.getOwnPropertySymbols(e);for(i=0;i<s.length;i++)n=s[i],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(a[n]=e[n])}return a}var o=i.createContext({}),d=function(e){var t=i.useContext(o),n=t;return e&&(n="function"==typeof e?e(t):l(l({},t),e)),n},p=function(e){var t=d(e.components);return i.createElement(o.Provider,{value:t},e.children)},k={inlineCode:"code",wrapper:function(e){var t=e.children;return i.createElement(i.Fragment,{},t)}},m=i.forwardRef((function(e,t){var n=e.components,a=e.mdxType,s=e.originalType,o=e.parentName,p=r(e,["components","mdxType","originalType","parentName"]),m=d(n),c=a,u=m["".concat(o,".").concat(c)]||m[c]||k[c]||s;return n?i.createElement(u,l(l({ref:t},p),{},{components:n})):i.createElement(u,l({ref:t},p))}));function c(e,t){var n=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var s=n.length,l=new Array(s);l[0]=m;var r={};for(var o in t)hasOwnProperty.call(t,o)&&(r[o]=t[o]);r.originalType=e,r.mdxType="string"==typeof e?e:a,l[1]=r;for(var d=2;d<s;d++)l[d]=n[d];return i.createElement.apply(null,l)}return i.createElement.apply(null,n)}m.displayName="MDXCreateElement"},9452:function(e,t,n){"use strict";n.r(t),n.d(t,{frontMatter:function(){return r},contentTitle:function(){return o},metadata:function(){return d},toc:function(){return p},default:function(){return m}});var i=n(2122),a=n(9756),s=(n(7294),n(3905)),l=["components"],r={},o=void 0,d={unversionedId:"tools/LISA_LPM/lisa-core/interfaces/fs.statsbase",id:"tools/LISA_LPM/lisa-core/interfaces/fs.statsbase",isDocsHomePage:!1,title:"fs.statsbase",description:"@listenai/lisa_core / Exports / fs / StatsBase",source:"@site/docs/tools/LISA_LPM/lisa-core/interfaces/fs.statsbase.md",sourceDirName:"tools/LISA_LPM/lisa-core/interfaces",slug:"/tools/LISA_LPM/lisa-core/interfaces/fs.statsbase",permalink:"/tools/LISA_LPM/lisa-core/interfaces/fs.statsbase",version:"current",frontMatter:{},sidebar:"toolsLISA",previous:{title:"fs.statoptions",permalink:"/tools/LISA_LPM/lisa-core/interfaces/fs.statoptions"},next:{title:"fs.writefileoptions",permalink:"/tools/LISA_LPM/lisa-core/interfaces/fs.writefileoptions"}},p=[{value:"Type parameters",id:"type-parameters",children:[]},{value:"Hierarchy",id:"hierarchy",children:[]},{value:"Table of contents",id:"table-of-contents",children:[{value:"Properties",id:"properties",children:[]},{value:"Methods",id:"methods",children:[]}]},{value:"Properties",id:"properties-1",children:[{value:"atime",id:"atime",children:[]},{value:"atimeMs",id:"atimems",children:[]},{value:"birthtime",id:"birthtime",children:[]},{value:"birthtimeMs",id:"birthtimems",children:[]},{value:"blksize",id:"blksize",children:[]},{value:"blocks",id:"blocks",children:[]},{value:"ctime",id:"ctime",children:[]},{value:"ctimeMs",id:"ctimems",children:[]},{value:"dev",id:"dev",children:[]},{value:"gid",id:"gid",children:[]},{value:"ino",id:"ino",children:[]},{value:"mode",id:"mode",children:[]},{value:"mtime",id:"mtime",children:[]},{value:"mtimeMs",id:"mtimems",children:[]},{value:"nlink",id:"nlink",children:[]},{value:"rdev",id:"rdev",children:[]},{value:"size",id:"size",children:[]},{value:"uid",id:"uid",children:[]}]},{value:"Methods",id:"methods-1",children:[{value:"isBlockDevice",id:"isblockdevice",children:[]},{value:"isCharacterDevice",id:"ischaracterdevice",children:[]},{value:"isDirectory",id:"isdirectory",children:[]},{value:"isFIFO",id:"isfifo",children:[]},{value:"isFile",id:"isfile",children:[]},{value:"isSocket",id:"issocket",children:[]},{value:"isSymbolicLink",id:"issymboliclink",children:[]}]}],k={toc:p};function m(e){var t=e.components,n=(0,a.Z)(e,l);return(0,s.kt)("wrapper",(0,i.Z)({},k,n,{components:t,mdxType:"MDXLayout"}),(0,s.kt)("p",null,(0,s.kt)("a",{parentName:"p",href:"/tools/LISA_LPM/lisa-core/README"},"@listenai/lisa_core")," / ",(0,s.kt)("a",{parentName:"p",href:"/tools/LISA_LPM/lisa-core/modules"},"Exports")," / ",(0,s.kt)("a",{parentName:"p",href:"/tools/LISA_LPM/lisa-core/modules/fs"},"fs")," / StatsBase"),(0,s.kt)("h1",{id:"interface-statsbaset"},"Interface: StatsBase<T",">"),(0,s.kt)("p",null,(0,s.kt)("a",{parentName:"p",href:"/tools/LISA_LPM/lisa-core/modules/fs"},"fs"),".StatsBase"),(0,s.kt)("h2",{id:"type-parameters"},"Type parameters"),(0,s.kt)("table",null,(0,s.kt)("thead",{parentName:"table"},(0,s.kt)("tr",{parentName:"thead"},(0,s.kt)("th",{parentName:"tr",align:"left"},"Name"))),(0,s.kt)("tbody",{parentName:"table"},(0,s.kt)("tr",{parentName:"tbody"},(0,s.kt)("td",{parentName:"tr",align:"left"},(0,s.kt)("inlineCode",{parentName:"td"},"T"))))),(0,s.kt)("h2",{id:"hierarchy"},"Hierarchy"),(0,s.kt)("ul",null,(0,s.kt)("li",{parentName:"ul"},(0,s.kt)("p",{parentName:"li"},(0,s.kt)("strong",{parentName:"p"},(0,s.kt)("inlineCode",{parentName:"strong"},"StatsBase"))),(0,s.kt)("p",{parentName:"li"},"\u21b3 ",(0,s.kt)("a",{parentName:"p",href:"/tools/LISA_LPM/lisa-core/classes/fs.stats"},(0,s.kt)("inlineCode",{parentName:"a"},"Stats"))),(0,s.kt)("p",{parentName:"li"},"\u21b3 ",(0,s.kt)("a",{parentName:"p",href:"/tools/LISA_LPM/lisa-core/classes/fs.bigintstats"},(0,s.kt)("inlineCode",{parentName:"a"},"BigIntStats"))))),(0,s.kt)("h2",{id:"table-of-contents"},"Table of contents"),(0,s.kt)("h3",{id:"properties"},"Properties"),(0,s.kt)("ul",null,(0,s.kt)("li",{parentName:"ul"},(0,s.kt)("a",{parentName:"li",href:"/tools/LISA_LPM/lisa-core/interfaces/fs.statsbase#atime"},"atime")),(0,s.kt)("li",{parentName:"ul"},(0,s.kt)("a",{parentName:"li",href:"/tools/LISA_LPM/lisa-core/interfaces/fs.statsbase#atimems"},"atimeMs")),(0,s.kt)("li",{parentName:"ul"},(0,s.kt)("a",{parentName:"li",href:"/tools/LISA_LPM/lisa-core/interfaces/fs.statsbase#birthtime"},"birthtime")),(0,s.kt)("li",{parentName:"ul"},(0,s.kt)("a",{parentName:"li",href:"/tools/LISA_LPM/lisa-core/interfaces/fs.statsbase#birthtimems"},"birthtimeMs")),(0,s.kt)("li",{parentName:"ul"},(0,s.kt)("a",{parentName:"li",href:"/tools/LISA_LPM/lisa-core/interfaces/fs.statsbase#blksize"},"blksize")),(0,s.kt)("li",{parentName:"ul"},(0,s.kt)("a",{parentName:"li",href:"/tools/LISA_LPM/lisa-core/interfaces/fs.statsbase#blocks"},"blocks")),(0,s.kt)("li",{parentName:"ul"},(0,s.kt)("a",{parentName:"li",href:"/tools/LISA_LPM/lisa-core/interfaces/fs.statsbase#ctime"},"ctime")),(0,s.kt)("li",{parentName:"ul"},(0,s.kt)("a",{parentName:"li",href:"/tools/LISA_LPM/lisa-core/interfaces/fs.statsbase#ctimems"},"ctimeMs")),(0,s.kt)("li",{parentName:"ul"},(0,s.kt)("a",{parentName:"li",href:"/tools/LISA_LPM/lisa-core/interfaces/fs.statsbase#dev"},"dev")),(0,s.kt)("li",{parentName:"ul"},(0,s.kt)("a",{parentName:"li",href:"/tools/LISA_LPM/lisa-core/interfaces/fs.statsbase#gid"},"gid")),(0,s.kt)("li",{parentName:"ul"},(0,s.kt)("a",{parentName:"li",href:"/tools/LISA_LPM/lisa-core/interfaces/fs.statsbase#ino"},"ino")),(0,s.kt)("li",{parentName:"ul"},(0,s.kt)("a",{parentName:"li",href:"/tools/LISA_LPM/lisa-core/interfaces/fs.statsbase#mode"},"mode")),(0,s.kt)("li",{parentName:"ul"},(0,s.kt)("a",{parentName:"li",href:"/tools/LISA_LPM/lisa-core/interfaces/fs.statsbase#mtime"},"mtime")),(0,s.kt)("li",{parentName:"ul"},(0,s.kt)("a",{parentName:"li",href:"/tools/LISA_LPM/lisa-core/interfaces/fs.statsbase#mtimems"},"mtimeMs")),(0,s.kt)("li",{parentName:"ul"},(0,s.kt)("a",{parentName:"li",href:"/tools/LISA_LPM/lisa-core/interfaces/fs.statsbase#nlink"},"nlink")),(0,s.kt)("li",{parentName:"ul"},(0,s.kt)("a",{parentName:"li",href:"/tools/LISA_LPM/lisa-core/interfaces/fs.statsbase#rdev"},"rdev")),(0,s.kt)("li",{parentName:"ul"},(0,s.kt)("a",{parentName:"li",href:"/tools/LISA_LPM/lisa-core/interfaces/fs.statsbase#size"},"size")),(0,s.kt)("li",{parentName:"ul"},(0,s.kt)("a",{parentName:"li",href:"/tools/LISA_LPM/lisa-core/interfaces/fs.statsbase#uid"},"uid"))),(0,s.kt)("h3",{id:"methods"},"Methods"),(0,s.kt)("ul",null,(0,s.kt)("li",{parentName:"ul"},(0,s.kt)("a",{parentName:"li",href:"/tools/LISA_LPM/lisa-core/interfaces/fs.statsbase#isblockdevice"},"isBlockDevice")),(0,s.kt)("li",{parentName:"ul"},(0,s.kt)("a",{parentName:"li",href:"/tools/LISA_LPM/lisa-core/interfaces/fs.statsbase#ischaracterdevice"},"isCharacterDevice")),(0,s.kt)("li",{parentName:"ul"},(0,s.kt)("a",{parentName:"li",href:"/tools/LISA_LPM/lisa-core/interfaces/fs.statsbase#isdirectory"},"isDirectory")),(0,s.kt)("li",{parentName:"ul"},(0,s.kt)("a",{parentName:"li",href:"/tools/LISA_LPM/lisa-core/interfaces/fs.statsbase#isfifo"},"isFIFO")),(0,s.kt)("li",{parentName:"ul"},(0,s.kt)("a",{parentName:"li",href:"/tools/LISA_LPM/lisa-core/interfaces/fs.statsbase#isfile"},"isFile")),(0,s.kt)("li",{parentName:"ul"},(0,s.kt)("a",{parentName:"li",href:"/tools/LISA_LPM/lisa-core/interfaces/fs.statsbase#issocket"},"isSocket")),(0,s.kt)("li",{parentName:"ul"},(0,s.kt)("a",{parentName:"li",href:"/tools/LISA_LPM/lisa-core/interfaces/fs.statsbase#issymboliclink"},"isSymbolicLink"))),(0,s.kt)("h2",{id:"properties-1"},"Properties"),(0,s.kt)("h3",{id:"atime"},"atime"),(0,s.kt)("p",null,"\u2022 ",(0,s.kt)("strong",{parentName:"p"},"atime"),": ",(0,s.kt)("inlineCode",{parentName:"p"},"Date")),(0,s.kt)("h4",{id:"defined-in"},"Defined in"),(0,s.kt)("p",null,"node_modules/@types/node/fs.d.ts:52"),(0,s.kt)("hr",null),(0,s.kt)("h3",{id:"atimems"},"atimeMs"),(0,s.kt)("p",null,"\u2022 ",(0,s.kt)("strong",{parentName:"p"},"atimeMs"),": ",(0,s.kt)("inlineCode",{parentName:"p"},"T")),(0,s.kt)("h4",{id:"defined-in-1"},"Defined in"),(0,s.kt)("p",null,"node_modules/@types/node/fs.d.ts:48"),(0,s.kt)("hr",null),(0,s.kt)("h3",{id:"birthtime"},"birthtime"),(0,s.kt)("p",null,"\u2022 ",(0,s.kt)("strong",{parentName:"p"},"birthtime"),": ",(0,s.kt)("inlineCode",{parentName:"p"},"Date")),(0,s.kt)("h4",{id:"defined-in-2"},"Defined in"),(0,s.kt)("p",null,"node_modules/@types/node/fs.d.ts:55"),(0,s.kt)("hr",null),(0,s.kt)("h3",{id:"birthtimems"},"birthtimeMs"),(0,s.kt)("p",null,"\u2022 ",(0,s.kt)("strong",{parentName:"p"},"birthtimeMs"),": ",(0,s.kt)("inlineCode",{parentName:"p"},"T")),(0,s.kt)("h4",{id:"defined-in-3"},"Defined in"),(0,s.kt)("p",null,"node_modules/@types/node/fs.d.ts:51"),(0,s.kt)("hr",null),(0,s.kt)("h3",{id:"blksize"},"blksize"),(0,s.kt)("p",null,"\u2022 ",(0,s.kt)("strong",{parentName:"p"},"blksize"),": ",(0,s.kt)("inlineCode",{parentName:"p"},"T")),(0,s.kt)("h4",{id:"defined-in-4"},"Defined in"),(0,s.kt)("p",null,"node_modules/@types/node/fs.d.ts:46"),(0,s.kt)("hr",null),(0,s.kt)("h3",{id:"blocks"},"blocks"),(0,s.kt)("p",null,"\u2022 ",(0,s.kt)("strong",{parentName:"p"},"blocks"),": ",(0,s.kt)("inlineCode",{parentName:"p"},"T")),(0,s.kt)("h4",{id:"defined-in-5"},"Defined in"),(0,s.kt)("p",null,"node_modules/@types/node/fs.d.ts:47"),(0,s.kt)("hr",null),(0,s.kt)("h3",{id:"ctime"},"ctime"),(0,s.kt)("p",null,"\u2022 ",(0,s.kt)("strong",{parentName:"p"},"ctime"),": ",(0,s.kt)("inlineCode",{parentName:"p"},"Date")),(0,s.kt)("h4",{id:"defined-in-6"},"Defined in"),(0,s.kt)("p",null,"node_modules/@types/node/fs.d.ts:54"),(0,s.kt)("hr",null),(0,s.kt)("h3",{id:"ctimems"},"ctimeMs"),(0,s.kt)("p",null,"\u2022 ",(0,s.kt)("strong",{parentName:"p"},"ctimeMs"),": ",(0,s.kt)("inlineCode",{parentName:"p"},"T")),(0,s.kt)("h4",{id:"defined-in-7"},"Defined in"),(0,s.kt)("p",null,"node_modules/@types/node/fs.d.ts:50"),(0,s.kt)("hr",null),(0,s.kt)("h3",{id:"dev"},"dev"),(0,s.kt)("p",null,"\u2022 ",(0,s.kt)("strong",{parentName:"p"},"dev"),": ",(0,s.kt)("inlineCode",{parentName:"p"},"T")),(0,s.kt)("h4",{id:"defined-in-8"},"Defined in"),(0,s.kt)("p",null,"node_modules/@types/node/fs.d.ts:38"),(0,s.kt)("hr",null),(0,s.kt)("h3",{id:"gid"},"gid"),(0,s.kt)("p",null,"\u2022 ",(0,s.kt)("strong",{parentName:"p"},"gid"),": ",(0,s.kt)("inlineCode",{parentName:"p"},"T")),(0,s.kt)("h4",{id:"defined-in-9"},"Defined in"),(0,s.kt)("p",null,"node_modules/@types/node/fs.d.ts:43"),(0,s.kt)("hr",null),(0,s.kt)("h3",{id:"ino"},"ino"),(0,s.kt)("p",null,"\u2022 ",(0,s.kt)("strong",{parentName:"p"},"ino"),": ",(0,s.kt)("inlineCode",{parentName:"p"},"T")),(0,s.kt)("h4",{id:"defined-in-10"},"Defined in"),(0,s.kt)("p",null,"node_modules/@types/node/fs.d.ts:39"),(0,s.kt)("hr",null),(0,s.kt)("h3",{id:"mode"},"mode"),(0,s.kt)("p",null,"\u2022 ",(0,s.kt)("strong",{parentName:"p"},"mode"),": ",(0,s.kt)("inlineCode",{parentName:"p"},"T")),(0,s.kt)("h4",{id:"defined-in-11"},"Defined in"),(0,s.kt)("p",null,"node_modules/@types/node/fs.d.ts:40"),(0,s.kt)("hr",null),(0,s.kt)("h3",{id:"mtime"},"mtime"),(0,s.kt)("p",null,"\u2022 ",(0,s.kt)("strong",{parentName:"p"},"mtime"),": ",(0,s.kt)("inlineCode",{parentName:"p"},"Date")),(0,s.kt)("h4",{id:"defined-in-12"},"Defined in"),(0,s.kt)("p",null,"node_modules/@types/node/fs.d.ts:53"),(0,s.kt)("hr",null),(0,s.kt)("h3",{id:"mtimems"},"mtimeMs"),(0,s.kt)("p",null,"\u2022 ",(0,s.kt)("strong",{parentName:"p"},"mtimeMs"),": ",(0,s.kt)("inlineCode",{parentName:"p"},"T")),(0,s.kt)("h4",{id:"defined-in-13"},"Defined in"),(0,s.kt)("p",null,"node_modules/@types/node/fs.d.ts:49"),(0,s.kt)("hr",null),(0,s.kt)("h3",{id:"nlink"},"nlink"),(0,s.kt)("p",null,"\u2022 ",(0,s.kt)("strong",{parentName:"p"},"nlink"),": ",(0,s.kt)("inlineCode",{parentName:"p"},"T")),(0,s.kt)("h4",{id:"defined-in-14"},"Defined in"),(0,s.kt)("p",null,"node_modules/@types/node/fs.d.ts:41"),(0,s.kt)("hr",null),(0,s.kt)("h3",{id:"rdev"},"rdev"),(0,s.kt)("p",null,"\u2022 ",(0,s.kt)("strong",{parentName:"p"},"rdev"),": ",(0,s.kt)("inlineCode",{parentName:"p"},"T")),(0,s.kt)("h4",{id:"defined-in-15"},"Defined in"),(0,s.kt)("p",null,"node_modules/@types/node/fs.d.ts:44"),(0,s.kt)("hr",null),(0,s.kt)("h3",{id:"size"},"size"),(0,s.kt)("p",null,"\u2022 ",(0,s.kt)("strong",{parentName:"p"},"size"),": ",(0,s.kt)("inlineCode",{parentName:"p"},"T")),(0,s.kt)("h4",{id:"defined-in-16"},"Defined in"),(0,s.kt)("p",null,"node_modules/@types/node/fs.d.ts:45"),(0,s.kt)("hr",null),(0,s.kt)("h3",{id:"uid"},"uid"),(0,s.kt)("p",null,"\u2022 ",(0,s.kt)("strong",{parentName:"p"},"uid"),": ",(0,s.kt)("inlineCode",{parentName:"p"},"T")),(0,s.kt)("h4",{id:"defined-in-17"},"Defined in"),(0,s.kt)("p",null,"node_modules/@types/node/fs.d.ts:42"),(0,s.kt)("h2",{id:"methods-1"},"Methods"),(0,s.kt)("h3",{id:"isblockdevice"},"isBlockDevice"),(0,s.kt)("p",null,"\u25b8 ",(0,s.kt)("strong",{parentName:"p"},"isBlockDevice"),"(): ",(0,s.kt)("inlineCode",{parentName:"p"},"boolean")),(0,s.kt)("h4",{id:"returns"},"Returns"),(0,s.kt)("p",null,(0,s.kt)("inlineCode",{parentName:"p"},"boolean")),(0,s.kt)("h4",{id:"defined-in-18"},"Defined in"),(0,s.kt)("p",null,"node_modules/@types/node/fs.d.ts:32"),(0,s.kt)("hr",null),(0,s.kt)("h3",{id:"ischaracterdevice"},"isCharacterDevice"),(0,s.kt)("p",null,"\u25b8 ",(0,s.kt)("strong",{parentName:"p"},"isCharacterDevice"),"(): ",(0,s.kt)("inlineCode",{parentName:"p"},"boolean")),(0,s.kt)("h4",{id:"returns-1"},"Returns"),(0,s.kt)("p",null,(0,s.kt)("inlineCode",{parentName:"p"},"boolean")),(0,s.kt)("h4",{id:"defined-in-19"},"Defined in"),(0,s.kt)("p",null,"node_modules/@types/node/fs.d.ts:33"),(0,s.kt)("hr",null),(0,s.kt)("h3",{id:"isdirectory"},"isDirectory"),(0,s.kt)("p",null,"\u25b8 ",(0,s.kt)("strong",{parentName:"p"},"isDirectory"),"(): ",(0,s.kt)("inlineCode",{parentName:"p"},"boolean")),(0,s.kt)("h4",{id:"returns-2"},"Returns"),(0,s.kt)("p",null,(0,s.kt)("inlineCode",{parentName:"p"},"boolean")),(0,s.kt)("h4",{id:"defined-in-20"},"Defined in"),(0,s.kt)("p",null,"node_modules/@types/node/fs.d.ts:31"),(0,s.kt)("hr",null),(0,s.kt)("h3",{id:"isfifo"},"isFIFO"),(0,s.kt)("p",null,"\u25b8 ",(0,s.kt)("strong",{parentName:"p"},"isFIFO"),"(): ",(0,s.kt)("inlineCode",{parentName:"p"},"boolean")),(0,s.kt)("h4",{id:"returns-3"},"Returns"),(0,s.kt)("p",null,(0,s.kt)("inlineCode",{parentName:"p"},"boolean")),(0,s.kt)("h4",{id:"defined-in-21"},"Defined in"),(0,s.kt)("p",null,"node_modules/@types/node/fs.d.ts:35"),(0,s.kt)("hr",null),(0,s.kt)("h3",{id:"isfile"},"isFile"),(0,s.kt)("p",null,"\u25b8 ",(0,s.kt)("strong",{parentName:"p"},"isFile"),"(): ",(0,s.kt)("inlineCode",{parentName:"p"},"boolean")),(0,s.kt)("h4",{id:"returns-4"},"Returns"),(0,s.kt)("p",null,(0,s.kt)("inlineCode",{parentName:"p"},"boolean")),(0,s.kt)("h4",{id:"defined-in-22"},"Defined in"),(0,s.kt)("p",null,"node_modules/@types/node/fs.d.ts:30"),(0,s.kt)("hr",null),(0,s.kt)("h3",{id:"issocket"},"isSocket"),(0,s.kt)("p",null,"\u25b8 ",(0,s.kt)("strong",{parentName:"p"},"isSocket"),"(): ",(0,s.kt)("inlineCode",{parentName:"p"},"boolean")),(0,s.kt)("h4",{id:"returns-5"},"Returns"),(0,s.kt)("p",null,(0,s.kt)("inlineCode",{parentName:"p"},"boolean")),(0,s.kt)("h4",{id:"defined-in-23"},"Defined in"),(0,s.kt)("p",null,"node_modules/@types/node/fs.d.ts:36"),(0,s.kt)("hr",null),(0,s.kt)("h3",{id:"issymboliclink"},"isSymbolicLink"),(0,s.kt)("p",null,"\u25b8 ",(0,s.kt)("strong",{parentName:"p"},"isSymbolicLink"),"(): ",(0,s.kt)("inlineCode",{parentName:"p"},"boolean")),(0,s.kt)("h4",{id:"returns-6"},"Returns"),(0,s.kt)("p",null,(0,s.kt)("inlineCode",{parentName:"p"},"boolean")),(0,s.kt)("h4",{id:"defined-in-24"},"Defined in"),(0,s.kt)("p",null,"node_modules/@types/node/fs.d.ts:34"))}m.isMDXComponent=!0}}]);