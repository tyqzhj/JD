requirejs([
    'jquery'
], function ($) {
    "use strict";

    //添加代码块的工具栏
    function getJupyterElement() {

        let codeCell = null;
        let textCell = null;

        let interval = setInterval(() => {

            codeCell = $('.code_cell');
            textCell = $('.text_cell');

            if (codeCell.length > 0 || textCell.length > 0) {

                clearInterval(interval);

                let html = `<div class="code-tools">
                    <div class='tool-container'>
                        <span class="markdown-btn" data-tooltip="markdown模式">Markdown</span>
                        <span class="code-btn" data-tooltip="代码模式">Code</span>
                    </div>
                    <div class='tool-container add-cell'>
                        <span class='add-cell-before' data-tooltip="向上添加代码块">
                            <i class='fa fa-plus'></i>
                            <i class='fa fa-caret-up'></i>
                        </span>
                        <span class='add-cell-after' data-tooltip="向下添加代码块">
                            <i class='fa fa-plus'></i>
                            <i class='fa fa-caret-down'></i>
                        </span>
                    </div>
                    <div class='tool-container delete-cell'>
                        <span class='delete-cell-btn' data-tooltip="删除">
                            <i class="fa fa-trash"></i>
                        </span>
                    </div>
                </div>`

                let runCodeHtml = `<div class="run-code-btn">
                        <i class='fa fa-play'></i>
                    </div>`

                codeCell.each(function (element, index) {

                    if ($(this).find('.input').find('.run-code-btn').length <= 0) {
                        $(this).find('.input').append(runCodeHtml);
                    }

                    if ($(this).find('.code-tools').length <= 0) {
                        $(this).append(html);
                    }

                });

                textCell.each(function (element, index) {

                    if ($(this).find('.code-tools').length <= 0) {
                        $(this).append(html);
                    }

                });
                //确保能获取到值后再调用
                runCode(html, runCodeHtml);
                addCell(html, runCodeHtml);
                deleteCell();
                changeType(html);
                // downloadCodes();
                addTooltip();
                postInputMessage();
            }
        }, 300);

    }

    function addTooltip() {

        $('.code-tools .tool-container span').on('mouseenter', function () {

            let tooltipTitle = $(this).data('tooltip');
            let html = `<div class='btn-tooltip'>${tooltipTitle}<div>`;
            $(this).append(html);

            let rightPx = $(this).find('.btn-tooltip').width() / 2 - $(this).width() / 2;

            $(this).find('.btn-tooltip').css({
                right: - rightPx + 'px',
            })

        }).on('mouseleave', function () {

            $(this).find('.btn-tooltip').remove();

        })
    }

    function addRunCodeBtn() {

        let downloadCode = `<div class="download-code">
            <i class="fa fa-download"></i>
        </div>`

        $('#modal_indicator').before(downloadCode);
    }

    window.onbeforeunload = function (e) {
        return false;
    }

    // addRunCodeBtn();
    getJupyterElement();

    // 添加代码执行按钮
    function runCode(codeCellHtml, runCodeHtml) {

        $('.run-code-btn').off('click').on('click', function () {

            $('#run_cell_select_below').click();

            $(this).hide();

            let runCodeTime = null;
            runCodeTime = setTimeout(() => {
                clearTimeout(runCodeTime);
                $(this).show();
            }, 500)

            getJupyterElement();

        })
    }

    //add text_cell or code_cell
    function addCell(codeCellHtml, runCodeHtml) {


        $('.add-cell-before').off('click').on('click', function (e) {
            e.stopPropagation();

            $('#insert_cell_above').click();

            getJupyterElement();

        })

        $('.add-cell-after').off('click').on('click', function (e) {
            e.stopPropagation();

            $('#insert_cell_below').click();

            getJupyterElement();

        })

    }


    // delete code_cell or text_cell
    function deleteCell() {

        $('.delete-cell-btn').off('click').on('click', function () {

            $('#delete_cell').click();

            getJupyterElement();

        });
    }


    //change type of cell, MarkDown or Code
    function changeType(codeCellHtml) {
        $('.markdown-btn').off('click').on('click', function () {
            $('#to_markdown').click();
            getJupyterElement();

        })

        $('.code-btn').off('click').on('click', function () {

            $('#to_code').click();
            getJupyterElement();

        })
    }

    function downloadCodes() {

        $('.download-code').off('click').on('click', function () {

            $('#download_script a').click();
        })
    }

    function postInputMessage() {
        // 传送消息给父级窗口，以记录学习时间
        window.addEventListener('input', function(e) {
            window.parent.postMessage('KeyboardInput', '*')
        })
    }
})
