(self.webpackChunkls_docs_web=self.webpackChunkls_docs_web||[]).push([[5179],{3905:function(e,t,n){"use strict";n.d(t,{Zo:function(){return d},kt:function(){return m}});var a=n(67294);function r(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function i(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);t&&(a=a.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,a)}return n}function l(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?i(Object(n),!0).forEach((function(t){r(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):i(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function o(e,t){if(null==e)return{};var n,a,r=function(e,t){if(null==e)return{};var n,a,r={},i=Object.keys(e);for(a=0;a<i.length;a++)n=i[a],t.indexOf(n)>=0||(r[n]=e[n]);return r}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(a=0;a<i.length;a++)n=i[a],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(r[n]=e[n])}return r}var p=a.createContext({}),s=function(e){var t=a.useContext(p),n=t;return e&&(n="function"==typeof e?e(t):l(l({},t),e)),n},d=function(e){var t=s(e.components);return a.createElement(p.Provider,{value:t},e.children)},c={inlineCode:"code",wrapper:function(e){var t=e.children;return a.createElement(a.Fragment,{},t)}},u=a.forwardRef((function(e,t){var n=e.components,r=e.mdxType,i=e.originalType,p=e.parentName,d=o(e,["components","mdxType","originalType","parentName"]),u=s(n),m=r,k=u["".concat(p,".").concat(m)]||u[m]||c[m]||i;return n?a.createElement(k,l(l({ref:t},d),{},{components:n})):a.createElement(k,l({ref:t},d))}));function m(e,t){var n=arguments,r=t&&t.mdxType;if("string"==typeof e||r){var i=n.length,l=new Array(i);l[0]=u;var o={};for(var p in t)hasOwnProperty.call(t,p)&&(o[p]=t[p]);o.originalType=e,o.mdxType="string"==typeof e?e:r,l[1]=o;for(var s=2;s<i;s++)l[s]=n[s];return a.createElement.apply(null,l)}return a.createElement.apply(null,n)}u.displayName="MDXCreateElement"},99696:function(e,t,n){"use strict";n.r(t),n.d(t,{frontMatter:function(){return o},contentTitle:function(){return p},metadata:function(){return s},toc:function(){return d},default:function(){return u}});var a=n(22122),r=n(19756),i=(n(67294),n(3905)),l=["components"],o={},p="GDB\u8c03\u8bd5",s={unversionedId:"chips/600X/gdbdebug/gdbdebug",id:"chips/600X/gdbdebug/gdbdebug",isDocsHomePage:!1,title:"GDB\u8c03\u8bd5",description:"\u6982\u8ff0",source:"@site/docs/chips/600X/gdbdebug/gdbdebug.md",sourceDirName:"chips/600X/gdbdebug",slug:"/chips/600X/gdbdebug/gdbdebug",permalink:"/chips/600X/gdbdebug/gdbdebug",version:"current",frontMatter:{},sidebar:"docs/chips/600X",previous:{title:"\u97f3\u9891\u64ad\u653e",permalink:"/chips/600X/application/modules/audio_playback"},next:{title:"SDK \u64cd\u4f5c",permalink:"/chips/600X/tool/sdk_command"}},d=[{value:"\u6982\u8ff0",id:"\u6982\u8ff0",children:[]},{value:"\u51c6\u5907\u5de5\u4f5c",id:"\u51c6\u5907\u5de5\u4f5c",children:[]},{value:"\u8c03\u8bd5\u8fc7\u7a0b",id:"\u8c03\u8bd5\u8fc7\u7a0b",children:[{value:"VS Code\u8c03\u8bd5\u73af\u5883\u642d\u5efa",id:"vs-code\u8c03\u8bd5\u73af\u5883\u642d\u5efa",children:[]},{value:"\u5bf9<code>hello_world</code>\u9879\u76ee\u8fdb\u884cdebug\u8c03\u8bd5",id:"\u5bf9hello_world\u9879\u76ee\u8fdb\u884cdebug\u8c03\u8bd5",children:[]}]}],c={toc:d};function u(e){var t=e.components,o=(0,r.Z)(e,l);return(0,i.kt)("wrapper",(0,a.Z)({},c,o,{components:t,mdxType:"MDXLayout"}),(0,i.kt)("h1",{id:"gdb\u8c03\u8bd5"},"GDB\u8c03\u8bd5"),(0,i.kt)("h2",{id:"\u6982\u8ff0"},"\u6982\u8ff0"),(0,i.kt)("p",null,"Zephyr\u63d0\u4f9b\u4e86\u591a\u79cddebug\u8c03\u8bd5\u65b9\u5f0f\uff0c\u652f\u6301GDB\u8c03\u8bd5\uff0c\u672c\u7ae0\u8282\u4e3b\u8981\u4ecb\u7ecd\u5982\u4f55\u4f7f\u7528VS Code\u548cJ-link\u4eff\u771f\u5668\u5b9e\u73b0\u5e94\u7528\u8c03\u8bd5\u3002\u901a\u8fc7\u672c\u7ae0\u8282\u5b66\u4e60\uff0c\u5f00\u53d1\u8005\u53ef\u4ee5\u4e86\u89e3\u5230\uff1a"),(0,i.kt)("ul",null,(0,i.kt)("li",{parentName:"ul"},"Windows\u4e0b\u5982\u4f55\u8fdb\u884c\u5e94\u7528\u7a0b\u5e8f\u7684\u8c03\u8bd5"),(0,i.kt)("li",{parentName:"ul"},"\u4f7f\u7528J-Link\u8c03\u8bd5CSK6\u82af\u7247")),(0,i.kt)("div",{className:"admonition admonition-note alert alert--secondary"},(0,i.kt)("div",{parentName:"div",className:"admonition-heading"},(0,i.kt)("h5",{parentName:"div"},(0,i.kt)("span",{parentName:"h5",className:"admonition-icon"},(0,i.kt)("svg",{parentName:"span",xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"},(0,i.kt)("path",{parentName:"svg",fillRule:"evenodd",d:"M6.3 5.69a.942.942 0 0 1-.28-.7c0-.28.09-.52.28-.7.19-.18.42-.28.7-.28.28 0 .52.09.7.28.18.19.28.42.28.7 0 .28-.09.52-.28.7a1 1 0 0 1-.7.3c-.28 0-.52-.11-.7-.3zM8 7.99c-.02-.25-.11-.48-.31-.69-.2-.19-.42-.3-.69-.31H6c-.27.02-.48.13-.69.31-.2.2-.3.44-.31.69h1v3c.02.27.11.5.31.69.2.2.42.31.69.31h1c.27 0 .48-.11.69-.31.2-.19.3-.42.31-.69H8V7.98v.01zM7 2.3c-3.14 0-5.7 2.54-5.7 5.68 0 3.14 2.56 5.7 5.7 5.7s5.7-2.55 5.7-5.7c0-3.15-2.56-5.69-5.7-5.69v.01zM7 .98c3.86 0 7 3.14 7 7s-3.14 7-7 7-7-3.12-7-7 3.14-7 7-7z"}))),"note")),(0,i.kt)("div",{parentName:"div",className:"admonition-content"},(0,i.kt)("p",{parentName:"div"},"Windows\u7cfb\u7edf\u548cUbuntu\u7cfb\u7edf\u4e0b\u914d\u7f6e\u7684\u57fa\u672c\u4e00\u81f4\uff0c"))),(0,i.kt)("h2",{id:"\u51c6\u5907\u5de5\u4f5c"},"\u51c6\u5907\u5de5\u4f5c"),(0,i.kt)("ul",null,(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("inlineCode",{parentName:"li"},"csk6-nanokit"),"\u5f00\u53d1\u677f"),(0,i.kt)("li",{parentName:"ul"},"J-link\u4eff\u771f\u5668"),(0,i.kt)("li",{parentName:"ul"},"Windows\u4e0b\u5b89\u88c5VS Code"),(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("inlineCode",{parentName:"li"},"csk6002_9s_nano"),"\u7684\u65e5\u5fd7\u4e32\u53e3",(0,i.kt)("inlineCode",{parentName:"li"},"A03 TX A02 RX"),"\u63a5\u4e32\u53e3\u677f\u8fde\u63a5\u7535\u8111\uff0c\u5728\u7535\u8111\u7aef\u4f7f\u7528\u4e32\u53e3\u8c03\u8bd5\u52a9\u624b\u67e5\u770b\u65e5\u5fd7\uff0c\u6ce2\u7279\u7387\u4e3a115200\u3002")),(0,i.kt)("p",null,"\u5c06",(0,i.kt)("inlineCode",{parentName:"p"},"csk6-nanokit"),"\u5f00\u53d1\u677f\u548cJ-link\u4eff\u771f\u5668SWD\u63a5\u53e3\u6309\u4ee5\u4e0b\u63a5\u7ebf\u56fe\u5b8c\u6574\u8fde\u63a5\uff1a\n",(0,i.kt)("img",{src:n(36604).Z})),(0,i.kt)("h2",{id:"\u8c03\u8bd5\u8fc7\u7a0b"},"\u8c03\u8bd5\u8fc7\u7a0b"),(0,i.kt)("h3",{id:"vs-code\u8c03\u8bd5\u73af\u5883\u642d\u5efa"},"VS Code\u8c03\u8bd5\u73af\u5883\u642d\u5efa"),(0,i.kt)("h4",{id:"vs-code\u5b89\u88c5venuss-cortex-debug\u8c03\u8bd5\u63d2\u4ef6"},"VS Code\u5b89\u88c5",(0,i.kt)("inlineCode",{parentName:"h4"},"Venus's Cortex-Debug"),"\u8c03\u8bd5\u63d2\u4ef6"),(0,i.kt)("p",null,"\u5728VS Code\u5e94\u7528\u5546\u5e97\u4e2d\u641c\u7d22",(0,i.kt)("inlineCode",{parentName:"p"},"Venus's Cortex-Debug"),"\u6269\u5c55\u63d2\u4ef6\uff0c\u5e76\u5b8c\u6210\u5b89\u88c5\uff0c\u5982\u4e0b\u56fe\u6240\u793a\uff1a"),(0,i.kt)("p",null,(0,i.kt)("img",{src:n(91643).Z})),(0,i.kt)("h4",{id:"\u521b\u5efahello_world\u9879\u76ee\u5e76\u5b8c\u6210debug\u73af\u5883\u914d\u7f6e"},"\u521b\u5efa",(0,i.kt)("inlineCode",{parentName:"h4"},"hello_world"),"\u9879\u76ee\u5e76\u5b8c\u6210debug\u73af\u5883\u914d\u7f6e"),(0,i.kt)("p",null,(0,i.kt)("strong",{parentName:"p"},"\u6b65\u9aa4\u4e00\uff1a")," \u5728Windows\u4e0b\u65b0\u5efa\u4e00\u4e2a\u6587\u4ef6\u5939\uff0c\u5982 ",(0,i.kt)("inlineCode",{parentName:"p"},"csk6-develop"),"\uff0c\u5e76\u5c06\u5176\u5bfc\u5165VS Code\u3002\n",(0,i.kt)("strong",{parentName:"p"},"\u6b65\u9aa4\u4e8c\uff1a")," \u5728",(0,i.kt)("inlineCode",{parentName:"p"},"csk6-develop"),"\u76ee\u5f55\u4e0b\u65b0\u5efa.vscode\u76ee\u5f55\uff0c\u5c06",(0,i.kt)("a",{parentName:"p",href:"https://iflyos-external.oss-cn-shanghai.aliyuncs.com/public/lsopen/zephyr/%E5%8A%9F%E8%83%BD%E6%96%87%E4%BB%B6/launch.json"},"launch.json"),"\u6587\u4ef6\u653e\u7f6e\u5230\u8be5\u76ee\u5f55\u4e0b\u3002\n",(0,i.kt)("strong",{parentName:"p"},"\u6b65\u9aa4\u4e09\uff1a")," \u5728",(0,i.kt)("inlineCode",{parentName:"p"},"csk6-develop"),"\u6587\u4ef6\u5939\u4e0b\u4f7f\u7528",(0,i.kt)("inlineCode",{parentName:"p"},"lisa zep create"),"\u547d\u4ee4\u521b\u5efa",(0,i.kt)("inlineCode",{parentName:"p"},"hello_world"),"\u9879\u76ee\uff0c\u5e76\u4f7f\u7528",(0,i.kt)("inlineCode",{parentName:"p"},"lisa zep build -b csk6002_9s_nano"),"\u547d\u4ee4\u5b8c\u6210\u7f16\u8bd1\u3002\n",(0,i.kt)("strong",{parentName:"p"},"\u6b65\u9aa4\u56db\uff1a")," \u4fee\u6539",(0,i.kt)("inlineCode",{parentName:"p"},"luncher.json"),"\u6587\u4ef6",(0,i.kt)("inlineCode",{parentName:"p"},"executable"),"\u3001",(0,i.kt)("inlineCode",{parentName:"p"},"serverpath"),"\u3001 ",(0,i.kt)("inlineCode",{parentName:"p"},"armToolchainPath"),"\u4e09\u9879\u914d\u7f6e\uff0c\u5177\u4f53\u5982\u4e0b\uff1a"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-c"},'# \u9700\u8981debug\u5e94\u7528\u7684zephyr.elf\u6587\u4ef6\u8def\u5f84\uff0c\u672c\u793a\u4f8b\u4e3a`hello_world`\u9879\u76ee\u7684zephyr.elf\u8def\u5f84\n "executable": "E:\\\\csk6-develop\\\\hello_world\\\\build\\\\zephyr\\\\zephyr.elf",\n\n# JLinkGDBServerCL.exe\u8def\u5f84\uff0clisa\u5b89\u88c5\u65f6\u5e26\u5165\uff0c\u9700\u8981\u4fee\u6539\u4e3a\u5bf9\u5e94\u7684\u8def\u5f84\n"serverpath": "C:/Users/xxxx/.listenai/lisa-zephyr/packages/node_modules/@binary/jlink-venus/binary/JLinkGDBServerCL.exe",\n\n# \u7f16\u8bd1\u94fe\u8def\u5f84\uff0clisa\u5b89\u88c5\u65f6\u5e26\u5165\uff0c\u9700\u8981\u4fee\u6539\u4e3a\u5bf9\u5e94\u7684\u8def\u5f84\n"armToolchainPath": "C:/Users/xxxx/.listenai/lisa-zephyr/packages/node_modules/@binary/gcc-arm-none-eabi-9/binary/bin",\n')),(0,i.kt)("p",null,(0,i.kt)("img",{src:n(11181).Z})),(0,i.kt)("div",{className:"admonition admonition-tip alert alert--success"},(0,i.kt)("div",{parentName:"div",className:"admonition-heading"},(0,i.kt)("h5",{parentName:"div"},(0,i.kt)("span",{parentName:"h5",className:"admonition-icon"},(0,i.kt)("svg",{parentName:"span",xmlns:"http://www.w3.org/2000/svg",width:"12",height:"16",viewBox:"0 0 12 16"},(0,i.kt)("path",{parentName:"svg",fillRule:"evenodd",d:"M6.5 0C3.48 0 1 2.19 1 5c0 .92.55 2.25 1 3 1.34 2.25 1.78 2.78 2 4v1h5v-1c.22-1.22.66-1.75 2-4 .45-.75 1-2.08 1-3 0-2.81-2.48-5-5.5-5zm3.64 7.48c-.25.44-.47.8-.67 1.11-.86 1.41-1.25 2.06-1.45 3.23-.02.05-.02.11-.02.17H5c0-.06 0-.13-.02-.17-.2-1.17-.59-1.83-1.45-3.23-.2-.31-.42-.67-.67-1.11C2.44 6.78 2 5.65 2 5c0-2.2 2.02-4 4.5-4 1.22 0 2.36.42 3.22 1.19C10.55 2.94 11 3.94 11 5c0 .66-.44 1.78-.86 2.48zM4 14h5c-.23 1.14-1.3 2-2.5 2s-2.27-.86-2.5-2z"}))),"tip")),(0,i.kt)("div",{parentName:"div",className:"admonition-content"},(0,i.kt)("p",{parentName:"div"},".vscode\u4e0b\u653e\u7f6e\u7684\u662f\u5404\u79cd\u914d\u7f6e\u6587\u4ef6\uff0c\u5982\uff1atask.json\u6587\u4ef6\u3001lauch.json\u6587\u4ef6\u7b49\uff0c.vscode\u6587\u4ef6\u5939\u901a\u5e38\u5728\u70b9\u51fb\u8fd0\u884c->\u542f\u52a8\u8c03\u8bd5\u540e\u7531\u7cfb\u7edf\u751f\u6210\uff0c\u672c\u793a\u4f8b\u76f4\u63a5\u65b0\u589e\u4e86\u8be5\u76ee\u5f55\u3002"))),(0,i.kt)("h3",{id:"\u5bf9hello_world\u9879\u76ee\u8fdb\u884cdebug\u8c03\u8bd5"},"\u5bf9",(0,i.kt)("inlineCode",{parentName:"h3"},"hello_world"),"\u9879\u76ee\u8fdb\u884cdebug\u8c03\u8bd5"),(0,i.kt)("h4",{id:"\u6b65\u9aa4\u4e00\uff1a\u57fa\u4e8ehello_world\u5e94\u7528\u4ee3\u7801\u589e\u52a0\u7b80\u5355\u7684\u4ee3\u7801\u903b\u8f91"},"\u6b65\u9aa4\u4e00\uff1a\u57fa\u4e8e",(0,i.kt)("inlineCode",{parentName:"h4"},"hello_world"),"\u5e94\u7528\u4ee3\u7801\u589e\u52a0\u7b80\u5355\u7684\u4ee3\u7801\u903b\u8f91"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-c"},'void main(void)\n{\n    printk("Hello World! %s\\n", CONFIG_BOARD);\n\n    /*\u589e\u52a0\u4e00\u4e2awhile\u5faa\u73af\u8ba1\u6570*/\n    int count = 0;\n    while (1) {\n        k_msleep(1000);\n        count++;\n        printk("count = %d", count);\n    }\n}\n')),(0,i.kt)("h4",{id:"\u6b65\u9aa4\u4e8c\uff1a\u7f16\u8bd1\u548c\u70e7\u5f55"},"\u6b65\u9aa4\u4e8c\uff1a\u7f16\u8bd1\u548c\u70e7\u5f55"),(0,i.kt)("ul",null,(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("strong",{parentName:"li"},"\u7f16\u8bd1"),"\n\u5728app\u6839\u76ee\u5f55\u4e0b\u901a\u8fc7\u4e00\u4e0b\u6307\u4ee4\u5b8c\u6210\u7f16\u8bd1\uff1a")),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre"},"lisa zep build -b csk6002_9s_nano\n")),(0,i.kt)("ul",null,(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("strong",{parentName:"li"},"\u70e7\u5f55"),"   ")),(0,i.kt)("p",null,"\u6307\u5b9aJlink\u4f5c\u4e3a\u70e7\u5f55\u65b9\u5f0f\u59cb\u70e7\u5f55\u56fa\u4ef6\uff1a"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre"},"lisa zep flash --runner jlink\n")),(0,i.kt)("div",{className:"admonition admonition-tip alert alert--success"},(0,i.kt)("div",{parentName:"div",className:"admonition-heading"},(0,i.kt)("h5",{parentName:"div"},(0,i.kt)("span",{parentName:"h5",className:"admonition-icon"},(0,i.kt)("svg",{parentName:"span",xmlns:"http://www.w3.org/2000/svg",width:"12",height:"16",viewBox:"0 0 12 16"},(0,i.kt)("path",{parentName:"svg",fillRule:"evenodd",d:"M6.5 0C3.48 0 1 2.19 1 5c0 .92.55 2.25 1 3 1.34 2.25 1.78 2.78 2 4v1h5v-1c.22-1.22.66-1.75 2-4 .45-.75 1-2.08 1-3 0-2.81-2.48-5-5.5-5zm3.64 7.48c-.25.44-.47.8-.67 1.11-.86 1.41-1.25 2.06-1.45 3.23-.02.05-.02.11-.02.17H5c0-.06 0-.13-.02-.17-.2-1.17-.59-1.83-1.45-3.23-.2-.31-.42-.67-.67-1.11C2.44 6.78 2 5.65 2 5c0-2.2 2.02-4 4.5-4 1.22 0 2.36.42 3.22 1.19C10.55 2.94 11 3.94 11 5c0 .66-.44 1.78-.86 2.48zM4 14h5c-.23 1.14-1.3 2-2.5 2s-2.27-.86-2.5-2z"}))),"tip")),(0,i.kt)("div",{parentName:"div",className:"admonition-content"},(0,i.kt)("p",{parentName:"div"},"\u5f85\u8c03\u8bd5\u7684\u5e94\u7528\u7a0b\u5e8f\u9700\u8981\u5b8c\u6210\u7f16\u8bd1\u5e76\u70e7\u5f55\u5230\u5f00\u53d1\u677f\u540e\u624d\u80fd\u8fdb\u884c\u8c03\u8bd5\u3002"))),(0,i.kt)("h4",{id:"\u6b65\u9aa4\u4e09\uff1a\u65ad\u70b9\u8c03\u8bd5"},"\u6b65\u9aa4\u4e09\uff1a\u65ad\u70b9\u8c03\u8bd5"),(0,i.kt)("p",null,(0,i.kt)("strong",{parentName:"p"},"\u589e\u52a0\u65ad\u70b9")),(0,i.kt)("p",null,"\u70b9\u51fbVC Code \u8fd0\u884c\u548c\u8c03\u8bd5\u6309\u94ae\u8fdb\u5165\u8c03\u8bd5\u6a21\u5f0f\uff0c\u5728main\u51fd\u6570\u7684\u7b2c12\u884c\u548c\u7b2c17\u884c\u5de6\u4fa7\u5355\u51fb\u9f20\u6807\u5de6\u952e\u589e\u52a0\u4e24\u4e2a\u65ad\u70b9\uff0c\u5982\u4e0b\u56fe\u793a\uff1a\n",(0,i.kt)("img",{src:n(71090).Z})),(0,i.kt)("p",null,(0,i.kt)("strong",{parentName:"p"},"\u5f00\u59cb\u8c03\u8bd5"),(0,i.kt)("br",{parentName:"p"}),"\n","\u589e\u52a0\u65ad\u70b9\u540e\u70b9\u51fb\u8fd0\u884c\u8c03\u8bd5\u6309\u94ae(F5)\u5f00\u59cb\u8fd0\u884c\uff0c\u53d8\u91cf\u503c\u53ca\u76d1\u63a7\u53d8\u91cf\u53ef\u4ee5\u5728\u8fd0\u884c\u8c03\u8bd5\u8fc7\u7a0b\u4e2d\u663e\u793a\uff0c\u5982\u4e0b\u56fe\u793a\uff1a"),(0,i.kt)("p",null,(0,i.kt)("img",{src:n(71005).Z})),(0,i.kt)("p",null,"\u4ece\u4e0a\u56fe\u53ef\u770b\u5230\uff0c\u5f53\u5f00\u59cbF5\u8fd0\u884c\u8c03\u8bd5\u540e\uff0c\u7a0b\u5e8f\u4f1a\u505c\u5728\u7b2c\u4e00\u4e2a\u65ad\u70b9\uff0c\u6b64\u65f6F5\u7ee7\u7eed\u6267\u884c\u540e\u7a0b\u5e8f\u8f93\u51fa",(0,i.kt)("inlineCode",{parentName:"p"},"Hello world\uff01"),"\u5e76\u8c03\u5230\u7b2c\u4e8c\u4e2a\u65ad\u70b9\u7b49\u5f85\u7ee7\u7eed\uff0c\u6b64\u540e\u6bcf\u6309\u4e00\u6b21F5\u7ee7\u7eed\u6267\u884c\u5219while\u5faa\u73af\u4e2dcount\u81ea\u589e\u5e76\u5faa\u73af\u8f93\u51fa\u65e5\u5fd7\u7b49\u5f85\u7ee7\u7eed\u6267\u884c\uff0c\u540c\u65f6\u5728\u4fa7\u8fb9\u680f\u53ef\u4ee5\u770b\u5230count\u5728\u8c03\u8bd5\u8fc7\u7a0b\u4e2d\u503c\u7684\u53d8\u5316\u3002"),(0,i.kt)("p",null,(0,i.kt)("strong",{parentName:"p"},"\u8c03\u8bd5\u64cd\u4f5c\u6309\u94ae\uff1a"),"\n",(0,i.kt)("img",{src:n(94526).Z})),(0,i.kt)("p",null,(0,i.kt)("strong",{parentName:"p"},"\u5e38\u7528\u8c03\u8bd5\u5feb\u6377\u952e\uff1a")),(0,i.kt)("ul",null,(0,i.kt)("li",{parentName:"ul"},"F5 \u7ee7\u7eed(\u6267\u884c\u5230\u4e0b\u4e00\u4e2a\u65ad\u70b9)"),(0,i.kt)("li",{parentName:"ul"},"F10 \u5355\u6b65\u8c03\u8fc7"),(0,i.kt)("li",{parentName:"ul"},"F11 \u5355\u6b65\u8c03\u8bd5"),(0,i.kt)("li",{parentName:"ul"},"Shift+F11 \u5355\u6b65\u8df3\u51fa"),(0,i.kt)("li",{parentName:"ul"},"ctrl+shift+F5 \u91cd\u542f"),(0,i.kt)("li",{parentName:"ul"},"shift+F5 \u505c\u6b62")),(0,i.kt)("p",null,"\u4ee5\u4e0a\u5c31\u662fZephyr\u5e94\u7528\u7a0b\u5e8f\u7684\u57fa\u672c\u8c03\u8bd5\u65b9\u6cd5\uff0c\u5f00\u53d1\u8005\u53ef\u4ee5\u5728\u4f7f\u7528\u8fc7\u7a0b\u4e2d\u5c1d\u8bd5\u66f4\u591a\u7684\u8c03\u8bd5\u624b\u6bb5\u3002"))}u.isMDXComponent=!0},36604:function(e,t,n){"use strict";t.Z=n.p+"assets/images/connect-b4a4002610747448d1332bbe01c2150f.png"},94526:function(e,t){"use strict";t.Z="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAUwAAABfCAYAAACDW/7eAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAAFiUAABYlAUlSJPAAAAz5SURBVHhe7d3pb1TXGcdx/wHs+77TfwAwqw22Ccbsa9khGAwY7Biw2cpqdogUkdBsbaIGkqhtAkpI6JKU0Ko0tGnpprQSalVVfZFUVZuqqPuSPJ3nOM/4zOG5M/fadzP+vfiIsX1m7rlH8tf3zp0ZisaO/QwBAEBhCCYAgE8IJgCATwgmAIBPCCYAgE8IJgCATwgmAIBPHSqYY8aMBYBOROtAklIVTG3BAAC8aB2JUuLB1BYBACAorS9hSyyY2g4DALSX1puwxB5MbQc1o0ePAQC4j9YLjdaf9oo1mNpOCW1hAAAK0XoitA61R2zB1HaGaQswevRoGjUKAOB+3AetG1pfmNajtoolmNpOsNwd1hdn1KhRxsiRANAZSQO0Prjx1DrDtC61RWLB9A6lvmi5RgLAA037vW/lBtQOp9YbrUttEXkwtcnrscxdkBEjRhYwAgAeSNrveyu7E3Y444hm7MHUYtm6APbCjKDhwwEAWtlBtY9E/URT61NQkQbTnXC+WMoi5C7QcBo2DABguOmB3YfWcMYXzUSCWSiWuQs1jIYOBYDOjDtgd6FwNDt4MGUHWo8u88fy/kUbSkOGAEBnwr/3bgvyRVOe0/Q6ytQ6FURRS7mj0bKzrYYMGWIMHjzYGDRokDFw4EBj3LhxAAC+STukJdIWaY3bIK1TQcQWTNkBr2AOGDBAXRAAAC/cjXzBZHaHtE4FkVgw7VgimADQFhJMN5p2b+wOaZ0KIvZgyl8AO5i80wgmAAQl7XCDaUfT7pDWqSAiC6Y9SYZgAkDYggaTab3yK9Fg8k5KMPv3768uCACAF+6GBFOiiWACACg6RTB5pySYvLMdIZiTJk2iaSUlVDp9Ok2fMSM0/Hj8uPz42nYBwJsEU6IpbUEwEzKhuJhKSkvV2IWNt8Pb0+YBAPdDMFOE4xX2EWUhvD1EE8AfBDNF4jqydPF2tfkAQK4HMpgy8Y4UTH5OUYtZXPCcJkBhfoLJ7B5pvfILwfTAF2K0kMWFt6/NCwBaIZgpEfdzly7evjavB1XJ7MW0ZO+TBt/WxiRlclklLW583ODb2hhIBoKZElrE4qbNK0rFk6bSgroztGz/s1S2aJ06JipzNn2OGt/6l8G3tTFJmPbQfNp66Ze05yYZfJu/p42F+CGYKaEFTCypO2F+sXd97R5VrarJ+Rl/PXf1tpzvtZU2r6AmFE+iuVuOUPXTt6nhjY/MnFn9lQ9ozblrVL5kY3bslLLZVPfK70wYljRdzHmcqKUxmG4sk4jmxKkTqWrNbJq7fg6Vzmp9mmbWioeo/ks7qPnWETr705N07uen6MxPTtDR7xyirRdrqGx+9H9wV1ZMpw+b6+lvZ3a1Cz8GP5a2jUIQzHYqLy+n2tpaqq+vN//y1/m+70ULmPAKZvUTN6jpxv+o8cZ/af2ZV6msfGbO/YLS5hXEgh2nqeHan+77hc/xzidU++JdKlu8AcG0eMVSxBVNDuXJHxwzQaw+/zCVVpbQzhfrs5H0cvrOcdpyYRONnzBefdwwSDDvnd5Jv2+uy6gPqM7cF8HMsCdoTzzqYFZWVtK+ffvo0KFD5l/+Ot/3vWgBE17BrL30fusvVSZE21+6S5VL1+bcNwhtXn6MHz+B1py+auLNc2l652Oqv/ohrT71qjndZg8/8e3MEeefs/Pd/c1/0NJ9TyOYGXYseQ23vvA+Nb79b4Nvy7rGEU07mNs+X0OH3z6QjeKp95pNPFcdWGHGbTi5jva+1mhiyT8/+7OT1PTKLppaNkV97PaSYN49uI0qpwXfBt+H74tgZtgTtCfeGYLZdPMTE0wTom/8nVbsu0gzZpTlPIYf2rwK4Viue/R6dvs7vvJbqlhe4zl24SPnzH7w2MZv/cdEgW9HHUx+qoC3L197BdMdFzU3liuPvURzNh9snVvmNn8vrmjawTz9o09DmDm6bLhUZ442tftULK6gA2/uMcFkDS/URXKkiWCGyJ6gPfGog5nkKbkEc9vlX9Ca5suZMf9s+cXLBLTmmdtUUbUg53EK0eZVSFX1fnO0yNutee7HNKkk/36yskXrzREo30dEGcyJU0po8xd+SBsu3DC3+XtaMPlnPIbHyriorT59xfyxkVhyrN258fey0cyM5ftojxUGO5gSy82PVatjbXxUuf9aU0toM0ecK/YuV8e1B4IZInuC9sSjDmZYtICJQsHkf/nrOWsygf70FJc1vPYHWrBpT3Z8Idq88uGr3HzKyNuqu/IBTav0fnkOXwWX03M5RedTd5lrlMFce/7NliPgDImmGyWJpYzj+2iPFTaO4dqzr9OKo5ezR7bu3GQcj+GxMi4KbjCbv3uYtjy+mWoe20TzNs5V7yNmLqmgozcPmvvtubo79KNMBDNE9gTtiXemYLLyWVW08cJb2VO4IBeEtHnlU/HZLbTr6381gVl+8Hl1jOAgShxdfGrOF4y0+4WBY8hHjXY05207lo0S37ZjGecRpkYLZlzmrKuiE7ePZoNp44tA2n1sHFc+LT/x/WMFAxsUghkie4L2xKMOZhpOye1gikXbDtPONz5qiVImAjte/hVVrdx83zibNq98FjY8mnnsj2nn9XvmNFsbI3gsz9/FF4KW7n1KvU+Y3Ghu//JvWp5DzeDbaYklSzKYbP3xtXT8e0fMkaaNL/Jo420cyZPvNZuXHK07ukYd01YIZojsCdoTjzqYabjoowWTzZy/lLY+d6clBnnGCW1e+cyrPU41X7xjTl8nTitTx6QJh7D6yVvZ9ciR+R7/LOlYsqSD2R7lC8uyp+V+jkiDQDBDZE/QnjiCGV0wOyI1mimKJUMwdQhmiOwJ2hOPOpid+ZS8o8qJZspiyTpyMHFKjmDGQguYSOtFn46MA7nwkfNGmmLJ0vLBIPxSoanlU9WfecFFHwQzFlrAhN9gxv2yonz4U3b4dZn80iG+sLLp6XfxyTsdBL+fnF+ozq+n5HfyTJ4+WR3nwsuKEMyC0nBKnuQL1zX8OsHqp97NhlvwFWh+7aZ2n7hNX7CaVp38qsG3tTFJScPc+AM1zDt38ML1HAhmO6Xhok+Sb43UlM5ZRg2v/zEnlmzn9b+kIk7L9j9jjnplXvxOpblbDqtj45aWuXEAD1zfawLI0ax/frs58tTG4q2RCKZvaQimkQlmUh++4UpzMPl92PxRc+7cal/+deIvjUrb3PgU2/7wDX4tZsPlOlq+e5l5frL67AZzVMkXeExYM7HEh28gmHkleUqeto93E2k+Jedgc7jduaUh5mmcG3/gBh8xShS94OPdEMxYaQETXsFk/HWaPkBYpPWiD44w20Y+QPj4rSPZePK/+ADhiIPZo0cPilL37t1zdOvWLatr165ZXbp0URckKfg/feKD5zChrbgbdkfsvrjt0foUVFHv3r0pKr169crRs2fPLJmAvUPagiQF/2tkvPgUl9/Dzh8Kku9TlpKQ5rl1dnY/pCl2Z9wGaZ0Koqhv374UlT59+uSQjdo7YMdTW5Ck4P8lB0g/O5J2V6Q1boO0TgVRJOf/UeDnF0S/fv2yZONuRLUFSVJJaakas6jxdrX5AEAuN47SFrs3doe0TgVRxE+URsXekD1pN5oSTm1BkjShuDj25zJ5e7xdbT4AkEtC6cbS7o3dIa1TQRTx1aQoyNUqIRt0A2rHU1uQpHG84jrS5O0glgD+aZF04+i2SOuVX0X2pfcwuRuSydrhtKPJtAVJC35OkS/EhH3EyY/Hj4vnLAGCk3bYsXRD6bZI65VfRfbrk8Lkbkgma4dT/hJINLUFAQDwIrGUlmihdFuk9cqv1ASTaQsCAOBF2hFbMLVXs4fB3ZBM1g4n75yEE8EEgKAkltISLZRui7Re+ZVoMCWa8tdBWxAAAC/SDollhw0msyeJYAJA2IIGU+tUELEHU6JpB1OiqS0IAIAXiaUbTLs3doe0TgWRWDDdaCKYABBUoaNLZndI61QQsQWT5Qsm0xYEAMCLtMMrmG6DtE4FkUgw80WT/2LY5GUDANC5uW0oFEvmNkjrVBCRBpO5E3aDqUVTCycAALM74caSRRVLllgwvaLphlNoCwcADz6tB3Yv7I7YfXHbo/UpqMiDydyJ2ztl76y9CEJbLADofLQ+2P2wu+I2R+tSWyQSTGbvnL3TTFsYAADhNsPuidYbrUttEUswmbYT9k4ydxFs2qIBwINP64FwG6J1RutRWyUaTObusNAWBwBA6wXT+sK0HrVVbMFk2s4IbQEAAArReiK0DrVHrMFk2k7ZtAUBAHBp/bBp/Wmv2IMptB0MQltAAOj4tN/3ILTehCWxYApthwEAgtL6ErbEgym0BQAAKETrSVRSE0yNtjgA0HlpnYhTqoMJAJAmCCYAgE8IJgCATwgmAIBPCCYAgE8IJgCATwgmAIBPCCYAgE8IJgCATwgmAIBPCCYAgE8IJgCATwgmAIAvw+j/8UcmEbpAOPoAAAAASUVORK5CYII="},91643:function(e,t,n){"use strict";t.Z=n.p+"assets/images/venus-debug-620dfef3c52acf95c769fe2ee3cbeb0e.png"},11181:function(e,t,n){"use strict";t.Z=n.p+"assets/images/venus-debug_config-dad1ec3c37ea0e05b846e14099d39993.png"},71090:function(e,t,n){"use strict";t.Z=n.p+"assets/images/venus-debug_rundebug-f9319c4c72e988bc4f479bdc4078d9d9.png"},71005:function(e,t,n){"use strict";t.Z=n.p+"assets/images/venus-debug_rundebug02-20c54515e9716d67fa935d6d7be690e1.png"}}]);