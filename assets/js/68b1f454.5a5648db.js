(self.webpackChunkls_docs_web=self.webpackChunkls_docs_web||[]).push([[2493],{3905:function(e,t,n){"use strict";n.d(t,{Zo:function(){return s},kt:function(){return f}});var r=n(67294);function a(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function i(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function l(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?i(Object(n),!0).forEach((function(t){a(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):i(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function p(e,t){if(null==e)return{};var n,r,a=function(e,t){if(null==e)return{};var n,r,a={},i=Object.keys(e);for(r=0;r<i.length;r++)n=i[r],t.indexOf(n)>=0||(a[n]=e[n]);return a}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(r=0;r<i.length;r++)n=i[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(a[n]=e[n])}return a}var o=r.createContext({}),c=function(e){var t=r.useContext(o),n=t;return e&&(n="function"==typeof e?e(t):l(l({},t),e)),n},s=function(e){var t=c(e.components);return r.createElement(o.Provider,{value:t},e.children)},u={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},d=r.forwardRef((function(e,t){var n=e.components,a=e.mdxType,i=e.originalType,o=e.parentName,s=p(e,["components","mdxType","originalType","parentName"]),d=c(n),f=a,h=d["".concat(o,".").concat(f)]||d[f]||u[f]||i;return n?r.createElement(h,l(l({ref:t},s),{},{components:n})):r.createElement(h,l({ref:t},s))}));function f(e,t){var n=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var i=n.length,l=new Array(i);l[0]=d;var p={};for(var o in t)hasOwnProperty.call(t,o)&&(p[o]=t[o]);p.originalType=e,p.mdxType="string"==typeof e?e:a,l[1]=p;for(var c=2;c<i;c++)l[c]=n[c];return r.createElement.apply(null,l)}return r.createElement.apply(null,n)}d.displayName="MDXCreateElement"},96795:function(e,t,n){"use strict";n.r(t),n.d(t,{frontMatter:function(){return p},contentTitle:function(){return o},metadata:function(){return c},toc:function(){return s},default:function(){return d}});var r=n(22122),a=n(19756),i=(n(67294),n(3905)),l=["components"],p={},o="\u83b7\u53d6/\u66f4\u65b0",c={unversionedId:"chips/600X/tool/lisa_plugin_zephyr/update",id:"chips/600X/tool/lisa_plugin_zephyr/update",isDocsHomePage:!1,title:"\u83b7\u53d6/\u66f4\u65b0",description:"\u83b7\u53d6",source:"@site/docs/chips/600X/tool/lisa_plugin_zephyr/update.md",sourceDirName:"chips/600X/tool/lisa_plugin_zephyr",slug:"/chips/600X/tool/lisa_plugin_zephyr/update",permalink:"/chips/600X/tool/lisa_plugin_zephyr/update",version:"current",frontMatter:{},sidebar:"docs/chips/600X",previous:{title:"\u7591\u96be\u89e3\u7b54",permalink:"/chips/600X/tool/lisa_plugin_zephyr/trouble_shooting"},next:{title:"\u7248\u672c\u66f4\u65b0\u65e5\u5fd7",permalink:"/chips/600X/tool/lisa_plugin_zephyr/release_note"}},s=[{value:"\u83b7\u53d6",id:"\u83b7\u53d6",children:[]},{value:"\u66f4\u65b0",id:"\u66f4\u65b0",children:[]},{value:"\u67e5\u770b\u5f53\u524d Lisa\u63d2\u4ef6-Zephyr \u7248\u672c",id:"\u67e5\u770b\u5f53\u524d-lisa\u63d2\u4ef6-zephyr-\u7248\u672c",children:[]}],u={toc:s};function d(e){var t=e.components,p=(0,a.Z)(e,l);return(0,i.kt)("wrapper",(0,r.Z)({},u,p,{components:t,mdxType:"MDXLayout"}),(0,i.kt)("h1",{id:"\u83b7\u53d6\u66f4\u65b0"},"\u83b7\u53d6/\u66f4\u65b0"),(0,i.kt)("h2",{id:"\u83b7\u53d6"},"\u83b7\u53d6"),(0,i.kt)("p",null,"\u53ef\u67e5\u770b ",(0,i.kt)("a",{parentName:"p",href:"../../quick_start/start_env"},"\u5feb\u901f\u4e0a\u624b/\u5b89\u88c5\u5f00\u53d1\u5de5\u5177")," \u7ae0\u8282\uff0c\u8fdb\u884c\u9996\u6b21\u5de5\u5177\u7684\u83b7\u53d6\u3002"),(0,i.kt)("h2",{id:"\u66f4\u65b0"},"\u66f4\u65b0"),(0,i.kt)("h4",{id:"\u66f4\u65b0\u5230-latest-\u7248\u672c"},"\u66f4\u65b0\u5230 ",(0,i.kt)("inlineCode",{parentName:"h4"},"latest")," \u7248\u672c"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-bash"},"lisa update zephyr\n")),(0,i.kt)("p",null,"\u6267\u884c\u8be5\u547d\u4ee4\u53ef\u5c06 Lisa\u63d2\u4ef6-Zephyr \u66f4\u65b0\u5230\u6700\u65b0\u7684",(0,i.kt)("inlineCode",{parentName:"p"},"latest"),"\u7248\u672c\u3002"),(0,i.kt)("h4",{id:"\u66f4\u65b0\u5230-beta-\u7248\u672c"},"\u66f4\u65b0\u5230 ",(0,i.kt)("inlineCode",{parentName:"h4"},"beta")," \u7248\u672c"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-bash"},"lisa install @lisa-plugin/zephyr@beta -g\n")),(0,i.kt)("p",null,"\u6267\u884c\u8be5\u547d\u4ee4\u53ef\u5c06 Lisa\u63d2\u4ef6-Zephyr \u66f4\u65b0\u5230\u6700\u65b0\u7684",(0,i.kt)("inlineCode",{parentName:"p"},"beta"),"\u7248\u672c\u3002"),(0,i.kt)("h4",{id:"\u66f4\u65b0\u5230-\u6307\u5b9a-\u7248\u672c"},"\u66f4\u65b0\u5230 ",(0,i.kt)("inlineCode",{parentName:"h4"},"\u6307\u5b9a")," \u7248\u672c"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-bash"},"lisa install @lisa-plugin/zephyr@1.6.0 -g\n")),(0,i.kt)("p",null,"\u6267\u884c\u8be5\u547d\u4ee4\u53ef\u5c06 Lisa\u63d2\u4ef6-Zephyr \u66f4\u65b0\u5230\u6700\u65b0\u7684",(0,i.kt)("inlineCode",{parentName:"p"},"\u6307\u5b9a\u76841.6.0")," \u7248\u672c\u3002"),(0,i.kt)("h2",{id:"\u67e5\u770b\u5f53\u524d-lisa\u63d2\u4ef6-zephyr-\u7248\u672c"},"\u67e5\u770b\u5f53\u524d Lisa\u63d2\u4ef6-Zephyr \u7248\u672c"),(0,i.kt)("p",null,"\u6267\u884c ",(0,i.kt)("inlineCode",{parentName:"p"},"lisa info zephyr"),"\uff0c\u5982\u4e0b\u56fe\u7ea2\u6846\u663e\u793a\u7684\u4e3a\u5f53\u524d Lisa\u63d2\u4ef6-Zephyr \u7248\u672c\u3002"),(0,i.kt)("p",null,(0,i.kt)("img",{alt:"image",src:n(1289).Z})))}d.isMDXComponent=!0},1289:function(e,t,n){"use strict";t.Z=n.p+"assets/images/update_1-8960817b3f14c8b1d3715aeff5aae5ad.png"}}]);