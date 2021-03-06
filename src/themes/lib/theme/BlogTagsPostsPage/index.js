"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _react = _interopRequireDefault(require("react"));

var _Layout = _interopRequireDefault(require("@theme/Layout"));

var _BlogPostItem = _interopRequireDefault(require("@theme/BlogPostItem"));

var _Link = _interopRequireDefault(require("@docusaurus/Link"));

var _BlogSidebar = _interopRequireDefault(require("@theme/BlogSidebar"));

var _Translate = _interopRequireWildcard(require("@docusaurus/Translate"));

var _themeCommon = require("@docusaurus/theme-common");

function _getRequireWildcardCache(nodeInterop) { if (typeof WeakMap !== "function") return null; var cacheBabelInterop = new WeakMap(); var cacheNodeInterop = new WeakMap(); return (_getRequireWildcardCache = function (nodeInterop) { return nodeInterop ? cacheNodeInterop : cacheBabelInterop; })(nodeInterop); }

function _interopRequireWildcard(obj, nodeInterop) { if (!nodeInterop && obj && obj.__esModule) { return obj; } if (obj === null || typeof obj !== "object" && typeof obj !== "function") { return { default: obj }; } var cache = _getRequireWildcardCache(nodeInterop); if (cache && cache.has(obj)) { return cache.get(obj); } var newObj = {}; var hasPropertyDescriptor = Object.defineProperty && Object.getOwnPropertyDescriptor; for (var key in obj) { if (key !== "default" && Object.prototype.hasOwnProperty.call(obj, key)) { var desc = hasPropertyDescriptor ? Object.getOwnPropertyDescriptor(obj, key) : null; if (desc && (desc.get || desc.set)) { Object.defineProperty(newObj, key, desc); } else { newObj[key] = obj[key]; } } } newObj.default = obj; if (cache) { cache.set(obj, newObj); } return newObj; }

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
// Very simple pluralization: probably good enough for now
function useBlogPostsPlural() {
  const {
    selectMessage
  } = (0, _themeCommon.usePluralForm)();
  return count => selectMessage(count, (0, _Translate.translate)({
    id: 'theme.blog.post.plurals',
    description: 'Pluralized label for "{count} posts". Use as much plural forms (separated by "|") as your language support (see https://www.unicode.org/cldr/cldr-aux/charts/34/supplemental/language_plural_rules.html)',
    message: 'One post|{count} posts'
  }, {
    count
  }));
}

function BlogTagsPostPage(props) {
  const {
    metadata,
    items,
    sidebar
  } = props;
  const {
    allTagsPath,
    name: tagName,
    count
  } = metadata;
  const blogPostsPlural = useBlogPostsPlural();
  const title = (0, _Translate.translate)({
    id: 'theme.blog.tagTitle',
    description: 'The title of the page for a blog tag',
    message: '{nPosts} tagged with "{tagName}"'
  }, {
    nPosts: blogPostsPlural(count),
    tagName
  });
  return <_Layout.default title={title} wrapperClassName={_themeCommon.ThemeClassNames.wrapper.blogPages} pageClassName={_themeCommon.ThemeClassNames.page.blogTagsPostPage} searchMetadatas={{
    // assign unique search tag to exclude this page from search results!
    tag: 'blog_tags_posts'
  }}>
      <div className="container margin-vert--lg">
        <div className="row">
          <aside className="col col--3">
            <_BlogSidebar.default sidebar={sidebar} />
          </aside>
          <main className="col col--7">
            <header className="margin-bottom--xl">
              <h1>{title}</h1>

              <_Link.default href={allTagsPath}>
                <_Translate.default id="theme.tags.tagsPageLink" description="The label of the link targeting the tag list page">
                  View All Tags
                </_Translate.default>
              </_Link.default>
            </header>

            {items.map(({
            content: BlogPostContent
          }) => <_BlogPostItem.default key={BlogPostContent.metadata.permalink} frontMatter={BlogPostContent.frontMatter} metadata={BlogPostContent.metadata} truncated>
                <BlogPostContent />
              </_BlogPostItem.default>)}
          </main>
        </div>
      </div>
    </_Layout.default>;
}

var _default = BlogTagsPostPage;
exports.default = _default;